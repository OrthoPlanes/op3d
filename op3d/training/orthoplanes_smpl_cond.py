import torch
from torch_utils import persistence
from training.network_stylegan2_multiplane import Generator as StyleGAN2Backbone
from training.volumetric_rendering.ray_sampler import RaySampler
import dnnlib
from training.volumetric_rendering import math_utils
from pytorch3d.ops import knn_points, knn_gather
from scipy.spatial.transform import Rotation
import numpy as np
# import mesh2sdf
from torch import nn
from pytorch3d.io import load_obj
from torch.nn import functional as F
import kaolin
from kaolin.ops.mesh import index_vertices_by_faces, check_sign
from kaolin.metrics.trianglemesh import point_to_mesh_distance
_Feature_dim = 32
_N = 6
_L = 2 * _N * 3 + 3

class Sdf2DensityRayMarcher(nn.Module):
    def __init__(self):
        super().__init__()

    def run_forward(self, colors, sdfs, depths, rendering_options, density_func):
        deltas = depths[:, :, 1:] - depths[:, :, :-1]
        deltas_inf = 1e9 * torch.ones_like(deltas[:, :, :1])
        deltas = torch.cat([deltas, deltas_inf], dim=-2)

        densities = density_func(sdfs)

        density_delta = densities * deltas

        alpha = 1 - torch.exp(-density_delta)

        alpha_shifted = torch.cat([torch.ones_like(alpha[:, :, :1]), 1-alpha + 1e-10], -2)
        weights = alpha * torch.cumprod(alpha_shifted, -2)[:, :, :-1]

        composite_rgb = torch.sum(weights * colors, -2)
        weight_total = weights.sum(2)
        weights_depth = weights.clone()
        weights_depth[:, :, -1] = weights_depth[:, :, -1] + (1 - weight_total)
        composite_depth = torch.sum(weights_depth * depths, -2) / weight_total

        # clip the composite to min/max range of depths
        composite_depth = torch.nan_to_num(composite_depth, float('inf'))
        composite_depth = torch.clamp(composite_depth, torch.min(depths), torch.max(depths))

        if True:
            composite_rgb = composite_rgb + 1 - weight_total

        composite_rgb = composite_rgb * 2 - 1 # Scale to (-1, 1)

        return composite_rgb, composite_depth, weights

    def forward(self, colors, densities, depths, rendering_options, density_func):
        composite_rgb, composite_depth, weights = self.run_forward(colors, densities, depths, rendering_options, density_func)

        return composite_rgb, composite_depth, weights


class Density(nn.Module):
    def __init__(self, params_init={}):
        super().__init__()
        for p in params_init:
            param = nn.Parameter(torch.tensor(params_init[p]))
            setattr(self, p, param)

    def forward(self, sdf, beta=None):
        return self.density_func(sdf, beta=beta)


class LaplaceDensity(Density):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    def __init__(self, params_init={}, beta_min=1e-4):
        super().__init__(params_init=params_init)
        self.beta_min = torch.tensor(beta_min).to(self.beta.device)

    def density_func(self, sdf, beta=None):
        if beta is None:
            beta = self.get_beta()

        alpha = 1 / beta
        # return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))
        return alpha * torch.sigmoid(-sdf / beta)
    def get_beta(self):
        beta = self.beta.abs() + self.beta_min
        return beta



class TriImportanceRenderer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ray_marcher = Sdf2DensityRayMarcher()
        self.deformation_decoder = DeformationDecoder()
        # mesh path
        self.mesh_file_path = '/home/hehonglin/nvme/projects/s2n_code/smpl_uv.obj'
        vertes, faces, aux = load_obj(self.mesh_file_path)
        self.face_idx = faces.verts_idx

    def forward(self, ws, planes, decoder, ray_origins, ray_directions, real_conditions, rendering_options, density_func):
        # self.plane_axes = self.plane_axes.to(ray_origins.device)

        if rendering_options['ray_start'] == rendering_options['ray_end'] == 'auto':
            # ray_start, ray_end = math_utils.get_ray_limits_box(ray_origins, ray_directions,
            #                                                    box_side_length=rendering_options['box_warp'])
            # is_ray_valid = ray_end > ray_start
            # if torch.any(is_ray_valid).item():
            #     ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
            #     ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
            depths_coarse = self.sample_stratified(ray_origins, 2.55, 2.95,
                                                   rendering_options['depth_resolution'],
                                                   rendering_options['disparity_space_sampling'])
        else:
            # Create stratified depth samples
            depths_coarse = self.sample_stratified(ray_origins, 2.55,
                                                   2.95, rendering_options['depth_resolution'],
                                                   rendering_options['disparity_space_sampling'])

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape

        # Coarse Pass
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(
            batch_size, -1, 3)
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)

        out, sdf_coarse, grad_coarse, delta_x_coarse = self.run_model(ws, planes, decoder, sample_coordinates, sample_directions, real_conditions, rendering_options)
        colors_coarse = out['rgb']
        densities_coarse = out['sigma']
        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)
        sdf_coarse = sdf_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)

        # Fine Pass
        N_importance = rendering_options['depth_resolution_importance']
        if N_importance > 0:
            densities_coarse = densities_coarse + sdf_coarse
            _, _, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options, density_func)

            depths_fine = self.sample_importance(depths_coarse, weights, N_importance)
            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
            sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(
                batch_size, -1, 3)

            out, sdf_fine, grad_fine, delta_x_fine = self.run_model(ws, planes, decoder, sample_coordinates, sample_directions, real_conditions, rendering_options)
            colors_fine = out['rgb']
            densities_fine = out['sigma']
            colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)
            sdf_fine = sdf_fine.reshape(batch_size, num_rays, N_importance, 1)

            all_depths, all_colors, all_densities, all_sdfs = self.unify_samples(depths_coarse, colors_coarse, densities_coarse, sdf_coarse,
                                                                       depths_fine, colors_fine, densities_fine, sdf_fine)

            all_densities = all_densities + all_sdfs

            # Aggregate
            rgb_final, depth_final, weights = self.ray_marcher(all_colors, all_densities, all_depths, rendering_options, density_func)

            sdf = {
                'sdf': all_densities,
                'sdf_zero': all_sdfs,
                'delta_x': torch.cat([delta_x_coarse, delta_x_fine], dim=-2),
                'eik_grad': torch.cat([grad_coarse, grad_fine], dim=-2) if grad_coarse is not None else None
            }
        else:
            densities_coarse = densities_coarse + sdf_coarse
            rgb_final, depth_final, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse,
                                                               rendering_options, density_func)
            sdf = {
                'sdf': densities_coarse,
                'sdf_zero': sdf_coarse,
                'delta_x': delta_x_coarse,
                'eik_grad': grad_coarse
            }

        return rgb_final, depth_final, weights.sum(2), sdf

    def run_model(self, ws, planes, decoder, sample_coordinates, sample_directions, real_conditions, options):
        sampled_features, coordinates, sdf, delta_x = self.sample_from_trimultiplanes(ws, planes, sample_coordinates, padding_mode='zeros',
                                                           real_conditions=real_conditions,
                                                           box_warp=options['box_warp'])
        out = decoder(sampled_features, None, None)
        if options.get('density_noise', 0) > 0:
            out['sigma'] += torch.randn_like(out['sigma']) * options['density_noise']

        if self.training:
            grad = torch.autograd.grad(outputs=out['sigma'],
                                       inputs=coordinates,
                                       grad_outputs=torch.ones_like(out['sigma']), create_graph=True)[0]
        else:
            grad = None

        return out, sdf, grad, delta_x

    def trans_to_tpose_coordinates(self, coordinates, real_conditions):
        batch_size, n_points, _ = coordinates.shape
        vertices = real_conditions['vertices'].to(coordinates.device)
        
        fk_matrices = real_conditions['fk_matrices'].to(coordinates.device)
        lbs_weights = real_conditions['lbs_weights'].to(coordinates.device)
        with torch.no_grad():
            n_vertices = vertices.shape[1]
            ik_matrices = torch.inverse(fk_matrices.float())
            vertex_ik_matrices = torch.einsum("bij,bjkl->bikl", lbs_weights.float(), ik_matrices.float())
            nearest_dists, nearest_indices, _ = knn_points(coordinates.float(), vertices.float())
            point_ik_matrices = knn_gather(vertex_ik_matrices.view(batch_size, n_vertices, 16), nearest_indices)
            point_ik_matrices = point_ik_matrices.mean(dim=2)
            point_ik_matrices = point_ik_matrices.reshape(batch_size, n_points, 4, 4)
        points_homo = torch.nn.functional.pad(coordinates, [0, 1], "constant", 1.)
        cano_coordinates = torch.einsum("bijk,bik->bij", point_ik_matrices, points_homo)
        cano_coordinates = cano_coordinates[:, :, :3]
        # scale parameters (to [-1, 1])
        cano_coordinates[..., 0] = (cano_coordinates[..., 0]) / 1.2
        cano_coordinates[..., 1] = (cano_coordinates[..., 1] + 0.3) / 1.2
        # points that far from the mesh will be rotated by root_rotation only
        rota_points = torch.einsum('bik, bnk->bni', torch.inverse(fk_matrices[:, 0].float()), points_homo)[..., :3]
        rota_points[..., 0] = rota_points[..., 0] / 1.2
        rota_points[..., 1] = (rota_points[..., 1] + 0.3) / 1.2
        
        cano_coordinates = torch.where(nearest_dists.mean(dim=2).reshape(batch_size, n_points, 1).repeat(1, 1, 3) < 0.1,
                                       cano_coordinates, rota_points)
        # sdf
        faces = self.face_idx.to(coordinates.device).to(torch.long)
        with torch.no_grad():
            face_vertices = index_vertices_by_faces(vertices, faces)
            distance, _, _ = point_to_mesh_distance(coordinates, face_vertices)
            sign_ = check_sign(vertices, faces, coordinates)
            sign_ = torch.where(sign_ == True, -1., 1.)
            original_sdf = distance * sign_
        return cano_coordinates, original_sdf.unsqueeze(-1)

    def sample_from_trimultiplanes(self, ws, plane_features, coordinates, mode='bilinear', padding_mode='zeros', real_conditions={},
                                   box_warp=None):
        assert padding_mode == 'zeros'
        N, n_planes_per_axis, C, H, W = plane_features.shape
        _, M, _ = coordinates.shape
        plane_features = plane_features.reshape(N, n_planes_per_axis, 3, C // 3, H, W)
        plane_features = plane_features.permute(0, 2, 3, 1, 4, 5).reshape(N, 3, C // 3, n_planes_per_axis, H, W)

        # todo: xyz -> canonical pose space
        coordinates = (2 / box_warp) * coordinates
        coordinates[..., 1] -= 0.15
        cano_coordinates, sdf = self.trans_to_tpose_coordinates(coordinates, real_conditions)
        cano_coordinates.requires_grad = True

        delta_x = self.deformation_decoder(cano_coordinates, ws, real_conditions)
        cano_coordinates = cano_coordinates + delta_x

        coordinates_XYZ = cano_coordinates.clone()[..., [0, 1, 2]]
        coordinates_YZX = cano_coordinates.clone()[..., [1, 2, 0]]
        coordinates_ZXY = cano_coordinates.clone()[..., [2, 0, 1]]

        output_features = torch.zeros([N, 3, coordinates.shape[1], _Feature_dim], device=coordinates.device)

        feature_XYZ = plane_features[:, 0]  # [N, C // 3, D, H, W]
        feature_YZX = plane_features[:, 1]  # [N, C // 3, D, H, W]
        feature_ZXY = plane_features[:, 2]  # [N, C // 3, D, H, W]

        sampled_coordinates_XYZ = coordinates_XYZ.unsqueeze(1).unsqueeze(2)
        sampled_coordinates_YZX = coordinates_YZX.unsqueeze(1).unsqueeze(2)
        sampled_coordinates_ZXY = coordinates_ZXY.unsqueeze(1).unsqueeze(2)

        output_features[:, 0, :, :] = self.grid_sample_3d(feature_XYZ, sampled_coordinates_XYZ.float()).permute(0, 4, 3,
                                                                                                                2,
                                                                                                                1).reshape(
            N, -1, _Feature_dim)
        output_features[:, 1, :, :] = self.grid_sample_3d(feature_YZX, sampled_coordinates_YZX.float()).permute(0, 4, 3,
                                                                                                                2,
                                                                                                                1).reshape(
            N, -1, _Feature_dim)
        output_features[:, 2, :, :] = self.grid_sample_3d(feature_ZXY, sampled_coordinates_ZXY.float()).permute(0, 4, 3,
                                                                                                                2,
                                                                                                                1).reshape(
            N, -1, _Feature_dim)

        return output_features, cano_coordinates, sdf, delta_x

    @staticmethod
    def grid_sample_3d(image, optical):
        N, C, ID, IH, IW = image.shape
        _, D, H, W, _ = optical.shape

        ix = optical[..., 0]
        iy = optical[..., 1]
        iz = optical[..., 2]

        ix = ((ix + 1) / 2) * (IW - 1)
        iy = ((iy + 1) / 2) * (IH - 1)
        iz = ((iz + 1) / 2) * (ID - 1)
        with torch.no_grad():
            ix_tnw = torch.floor(ix)
            iy_tnw = torch.floor(iy)
            iz_tnw = torch.floor(iz)

            ix_tne = ix_tnw + 1
            iy_tne = iy_tnw
            iz_tne = iz_tnw

            ix_tsw = ix_tnw
            iy_tsw = iy_tnw + 1
            iz_tsw = iz_tnw

            ix_tse = ix_tnw + 1
            iy_tse = iy_tnw + 1
            iz_tse = iz_tnw

            ix_bnw = ix_tnw
            iy_bnw = iy_tnw
            iz_bnw = iz_tnw + 1

            ix_bne = ix_tnw + 1
            iy_bne = iy_tnw
            iz_bne = iz_tnw + 1

            ix_bsw = ix_tnw
            iy_bsw = iy_tnw + 1
            iz_bsw = iz_tnw + 1

            ix_bse = ix_tnw + 1
            iy_bse = iy_tnw + 1
            iz_bse = iz_tnw + 1

        tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz)
        tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz)
        tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz)
        tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz)
        bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse)
        bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw)
        bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne)
        bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw)

        with torch.no_grad():
            torch.clamp(ix_tnw, 0, IW - 1, out=ix_tnw)
            torch.clamp(iy_tnw, 0, IH - 1, out=iy_tnw)
            torch.clamp(iz_tnw, 0, ID - 1, out=iz_tnw)

            torch.clamp(ix_tne, 0, IW - 1, out=ix_tne)
            torch.clamp(iy_tne, 0, IH - 1, out=iy_tne)
            torch.clamp(iz_tne, 0, ID - 1, out=iz_tne)

            torch.clamp(ix_tsw, 0, IW - 1, out=ix_tsw)
            torch.clamp(iy_tsw, 0, IH - 1, out=iy_tsw)
            torch.clamp(iz_tsw, 0, ID - 1, out=iz_tsw)

            torch.clamp(ix_tse, 0, IW - 1, out=ix_tse)
            torch.clamp(iy_tse, 0, IH - 1, out=iy_tse)
            torch.clamp(iz_tse, 0, ID - 1, out=iz_tse)

            torch.clamp(ix_bnw, 0, IW - 1, out=ix_bnw)
            torch.clamp(iy_bnw, 0, IH - 1, out=iy_bnw)
            torch.clamp(iz_bnw, 0, ID - 1, out=iz_bnw)

            torch.clamp(ix_bne, 0, IW - 1, out=ix_bne)
            torch.clamp(iy_bne, 0, IH - 1, out=iy_bne)
            torch.clamp(iz_bne, 0, ID - 1, out=iz_bne)

            torch.clamp(ix_bsw, 0, IW - 1, out=ix_bsw)
            torch.clamp(iy_bsw, 0, IH - 1, out=iy_bsw)
            torch.clamp(iz_bsw, 0, ID - 1, out=iz_bsw)

            torch.clamp(ix_bse, 0, IW - 1, out=ix_bse)
            torch.clamp(iy_bse, 0, IH - 1, out=iy_bse)
            torch.clamp(iz_bse, 0, ID - 1, out=iz_bse)

        image = image.reshape(N, C, ID * IH * IW)

        tnw_val = torch.gather(image, 2,
                               (iz_tnw * IW * IH + iy_tnw * IW + ix_tnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
        tne_val = torch.gather(image, 2,
                               (iz_tne * IW * IH + iy_tne * IW + ix_tne).long().view(N, 1, D * H * W).repeat(1, C, 1))
        tsw_val = torch.gather(image, 2,
                               (iz_tsw * IW * IH + iy_tsw * IW + ix_tsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
        tse_val = torch.gather(image, 2,
                               (iz_tse * IW * IH + iy_tse * IW + ix_tse).long().view(N, 1, D * H * W).repeat(1, C, 1))
        bnw_val = torch.gather(image, 2,
                               (iz_bnw * IW * IH + iy_bnw * IW + ix_bnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
        bne_val = torch.gather(image, 2,
                               (iz_bne * IW * IH + iy_bne * IW + ix_bne).long().view(N, 1, D * H * W).repeat(1, C, 1))
        bsw_val = torch.gather(image, 2,
                               (iz_bsw * IW * IH + iy_bsw * IW + ix_bsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
        bse_val = torch.gather(image, 2,
                               (iz_bse * IW * IH + iy_bse * IW + ix_bse).long().view(N, 1, D * H * W).repeat(1, C, 1))

        out_val = (tnw_val.view(N, C, D, H, W) * tnw.view(N, 1, D, H, W) +
                   tne_val.view(N, C, D, H, W) * tne.view(N, 1, D, H, W) +
                   tsw_val.view(N, C, D, H, W) * tsw.view(N, 1, D, H, W) +
                   tse_val.view(N, C, D, H, W) * tse.view(N, 1, D, H, W) +
                   bnw_val.view(N, C, D, H, W) * bnw.view(N, 1, D, H, W) +
                   bne_val.view(N, C, D, H, W) * bne.view(N, 1, D, H, W) +
                   bsw_val.view(N, C, D, H, W) * bsw.view(N, 1, D, H, W) +
                   bse_val.view(N, C, D, H, W) * bse.view(N, 1, D, H, W))

        return out_val

    def sort_samples(self, all_depths, all_colors, all_densities):
        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))
        return all_depths, all_colors, all_densities

    def unify_samples(self, depths1, colors1, densities1, sdf1, depths2, colors2, densities2, sdf2):
        all_depths = torch.cat([depths1, depths2], dim=-2)
        all_sdfs = torch.cat([sdf1, sdf2], dim=-2)
        all_colors = torch.cat([colors1, colors2], dim=-2)
        all_densities = torch.cat([densities1, densities2], dim=-2)

        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))
        all_sdfs = torch.gather(all_sdfs, -2, indices.expand(-1, -1, -1, 1))

        return all_depths, all_colors, all_densities, all_sdfs

    def sample_stratified(self, ray_origins, ray_start, ray_end, depth_resolution, disparity_space_sampling=False):
        """
        Return depths of approximately uniformly spaced samples along rays.
        """
        N, M, _ = ray_origins.shape
        if disparity_space_sampling:
            depths_coarse = torch.linspace(0,
                                           1,
                                           depth_resolution,
                                           device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1,
                                                                                                                1)
            depth_delta = 1 / (depth_resolution - 1)
            depths_coarse += torch.rand_like(depths_coarse) * depth_delta
            depths_coarse = 1. / (1. / ray_start * (1. - depths_coarse) + 1. / ray_end * depths_coarse)
        else:
            if type(ray_start) == torch.Tensor:
                depths_coarse = math_utils.linspace(ray_start, ray_end, depth_resolution).permute(1, 2, 0, 3)
                depth_delta = (ray_end - ray_start) / (depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta[..., None]
            else:
                depths_coarse = torch.linspace(ray_start, ray_end, depth_resolution, device=ray_origins.device).reshape(
                    1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
                depth_delta = (ray_end - ray_start) / (depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta

        return depths_coarse

    def sample_importance(self, z_vals, weights, N_importance):
        """
        Return depths of importance sampled points along rays. See NeRF importance sampling for more.
        """
        with torch.no_grad():
            batch_size, num_rays, samples_per_ray, _ = z_vals.shape

            z_vals = z_vals.reshape(batch_size * num_rays, samples_per_ray)
            weights = weights.reshape(batch_size * num_rays, -1)  # -1 to account for loss of 1 sample in MipRayMarcher

            # smooth weights
            weights = torch.nn.functional.max_pool1d(weights.unsqueeze(1).float(), 2, 1, padding=1)
            weights = torch.nn.functional.avg_pool1d(weights, 2, 1).squeeze()
            weights = weights + 0.01

            z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
            importance_z_vals = self.sample_pdf(z_vals_mid, weights[:, 1:-1],
                                                N_importance).detach().reshape(batch_size, num_rays, N_importance, 1)
        return importance_z_vals

    def sample_pdf(self, bins, weights, N_importance, det=False, eps=1e-5):
        """
        Sample @N_importance samples from @bins with distribution defined by @weights.
        Inputs:
            bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
            weights: (N_rays, N_samples_)
            N_importance: the number of samples to draw from the distribution
            det: deterministic or not
            eps: a small number to prevent division by zero
        Outputs:
            samples: the sampled samples
        """
        N_rays, N_samples_ = weights.shape
        weights = weights + eps  # prevent division by zero (don't do inplace op!)
        pdf = weights / torch.sum(weights, -1, keepdim=True)  # (N_rays, N_samples_)
        cdf = torch.cumsum(pdf, -1)  # (N_rays, N_samples), cumulative distribution function
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (N_rays, N_samples_+1)
        # padded to 0~1 inclusive

        if det:
            u = torch.linspace(0, 1, N_importance, device=bins.device)
            u = u.expand(N_rays, N_importance)
        else:
            u = torch.rand(N_rays, N_importance, device=bins.device)
        u = u.contiguous()

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp_min(inds - 1, 0)
        above = torch.clamp_max(inds, N_samples_)

        inds_sampled = torch.stack([below, above], -1).view(N_rays, 2 * N_importance)
        cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
        bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom[denom < eps] = 1  # denom equals 0 means a bin has weight 0, in which case it will not be sampled
        # anyway, therefore any value for it is fine (set to 1 here)

        samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (bins_g[..., 1] - bins_g[..., 0])
        return samples


@persistence.persistent_class
class TriMultiPlaneGenerator(torch.nn.Module):
    def __init__(self,
                 z_dim,  # Input latent (Z) dimensionality.
                 c_dim,  # Conditioning label (C) dimensionality.
                 w_dim,  # Intermediate latent (W) dimensionality.
                 img_resolution,  # Output resolution.
                 img_channels,  # Number of output color channels.
                 sr_num_fp16_res=0,
                 mapping_kwargs={},  # Arguments for MappingNetwork.
                 rendering_kwargs={},
                 sr_kwargs={},
                 planes=9,
                 **synthesis_kwargs,  # Arguments for SynthesisNetwork.
                 ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.renderer = TriImportanceRenderer()
        self.ray_sampler = RaySampler()
        self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=256, img_channels=_Feature_dim * 3,
                                          mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        self.superresolution = dnnlib.util.construct_class_by_name(
            class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution,
            sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        self.decoder = OSGDecoder(_Feature_dim, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1),
                                       'decoder_output_dim': 32})
        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs

        self._last_planes = None

        self.n_planes_per_axis = planes
        self.L = 10
        self.density_func = LaplaceDensity(params_init={'beta': 1e-2})

    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
            c = torch.zeros_like(c)
        return self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi,
                                     truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def synthesis(self, ws, c, real_conditions, neural_rendering_resolution=None, update_emas=False, cache_backbone=False,
                  use_cached_backbone=False, feature=False, **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # todo: clone the ws and -> [B, 3, N_planes_per_axis, D_w]
        # ws_cloned = ws[:, -1:, :].clone()
        ws_cloned = ws.clone()
        ws_cloned = ws_cloned.reshape(-1, 1, 13, ws.shape[-1]).expand(ws.shape[0],
                                                                  self.n_planes_per_axis, 13,
                                                                  ws.shape[-1])

        # generate coordinates embeedings
        useful_coordinate_point = torch.linspace(-1, 1, self.n_planes_per_axis).to(ws.device).\
            reshape(1, self.n_planes_per_axis, 1).\
            expand(ws.shape[0], self.n_planes_per_axis, 1)
        # coordinate based feature
        # coord_embedding = torch.cat([ws_cloned, useful_coordinate_point], dim=-1)
        useful_coordinate_point = useful_coordinate_point.reshape(ws.shape[0] * self.n_planes_per_axis)
        # coordinate based feature
        embedding = torch.zeros([useful_coordinate_point.shape[0], self.L * 2 + 1], device=useful_coordinate_point.device)
        embedding[..., 0] = useful_coordinate_point
        for i in range(self.L):
            embedding[..., 2 * i + 1] = torch.cos(2 ** i * torch.pi * useful_coordinate_point)
            embedding[..., 2 * i + 2] = torch.sin(2 ** i * torch.pi * useful_coordinate_point)
        # coordinate based feature
        embedding = embedding.reshape(ws.shape[0], self.n_planes_per_axis, 1, self.L * 2 + 1).repeat(1, 1, 13, 1)
        coord_embedding = torch.cat([ws_cloned, embedding], dim=-1)

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)
        rot_mat = Rotation.from_euler('xyz', [np.pi, 0, 0]).as_matrix()
        ray_origins = ray_origins @ torch.from_numpy(rot_mat).float().to(ray_origins.device)
        ray_directions = ray_directions @ torch.from_numpy(rot_mat).float().to(ray_origins.device)
        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, coord_embedding, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # Reshape output into three 32-channel planes
        planes = planes.view(ws.shape[0], self.n_planes_per_axis, planes.shape[-3], planes.shape[-2], planes.shape[-1])

        # Perform volume rendering
        feature_samples, depth_samples, weights_samples, sdf = self.renderer(ws[:, -1].clone(), planes, self.decoder, ray_origins,
                                                                        ray_directions,
                                                                        real_conditions,
                                                                        self.rendering_kwargs,
                                                                        self.density_func)  # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        f = feature_image if feature else None
        sr_image = self.superresolution(rgb_image, feature_image, ws,
                                        noise_mode=self.rendering_kwargs['superresolution_noise_mode'],
                                        **{k: synthesis_kwargs[k] for k in synthesis_kwargs.keys() if
                                           k != 'noise_mode'})

        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image, 'sdf': sdf, 'ff': f}

    def sample(self, coordinates, directions, ws, c, real_conditions, truncation_psi=1, truncation_cutoff=None, update_emas=False,
               **synthesis_kwargs):
        ws_cloned = ws.clone()
        ws_cloned = ws_cloned.reshape(-1, 1, 13, ws.shape[-1]).expand(ws.shape[0],
                                                                  self.n_planes_per_axis, 13,
                                                                  ws.shape[-1])

        # generate coordinates embeedings
        useful_coordinate_point = torch.linspace(-1, 1, self.n_planes_per_axis).to(ws.device).\
            reshape(1, self.n_planes_per_axis, 1).\
            expand(ws.shape[0], self.n_planes_per_axis, 1)
        # coordinate based feature
        # coord_embedding = torch.cat([ws_cloned, useful_coordinate_point], dim=-1)
        useful_coordinate_point = useful_coordinate_point.reshape(ws.shape[0] * self.n_planes_per_axis)
        # coordinate based feature
        embedding = torch.zeros([useful_coordinate_point.shape[0], self.L * 2 + 1], device=useful_coordinate_point.device)
        embedding[..., 0] = useful_coordinate_point
        for i in range(self.L):
            embedding[..., 2 * i + 1] = torch.cos(2 ** i * torch.pi * useful_coordinate_point)
            embedding[..., 2 * i + 2] = torch.sin(2 ** i * torch.pi * useful_coordinate_point)
        # coordinate based feature
        embedding = embedding.reshape(ws.shape[0], self.n_planes_per_axis, 1, self.L * 2 + 1).repeat(1, 1, 13, 1)
        coord_embedding = torch.cat([ws_cloned, embedding], dim=-1)
        planes = self.backbone.synthesis(ws, coord_embedding, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(ws.shape[0], self.n_planes_per_axis, planes.shape[-3], planes.shape[-2], planes.shape[-1])
        out, sdf, _, _ = self.renderer.run_model(ws[:, -1].clone(), planes, self.decoder, coordinates, directions, real_conditions, self.rendering_kwargs)
        out['sigma'] += sdf.reshape(out['sigma'].shape)
        out['sigma'] = self.density_func(out['sigma'])
        return out

    def sample_mixed(self, coordinates, directions, ws, real_conditions, truncation_psi=1, truncation_cutoff=None, update_emas=False,
                     **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        ws_cloned = ws[:, -1:, :].clone()
        ws_cloned = ws_cloned.reshape(-1, 1, ws.shape[-1]).expand(ws.shape[0],
                                                                     self.n_planes_per_axis,
                                                                     ws.shape[-1])

        # generate coordinates embeddings
        useful_coordinate_point = torch.linspace(-1, 1, self.n_planes_per_axis).to(ws.device). \
            reshape(1, self.n_planes_per_axis, 1). \
            expand(ws.shape[0], self.n_planes_per_axis, 1)
        # coordinate based feature
        # coord_embedding = torch.cat([ws_cloned, useful_coordinate_point], dim=-1)
        useful_coordinate_point = useful_coordinate_point.reshape(ws.shape[0] * self.n_planes_per_axis)
        # coordinate based feature
        embedding = torch.zeros([useful_coordinate_point.shape[0], self.L * 2 + 1], device=useful_coordinate_point.device)
        embedding[..., 0] = useful_coordinate_point
        for i in range(self.L):
            embedding[..., 2 * i + 1] = torch.cos(2 ** i * torch.pi * useful_coordinate_point)
            embedding[..., 2 * i + 2] = torch.sin(2 ** i * torch.pi * useful_coordinate_point)
        # coordinate based feature
        embedding = embedding.reshape(ws.shape[0], self.n_planes_per_axis, self.L * 2 + 1)
        coord_embedding = torch.cat([ws_cloned, embedding], dim=-1)
        planes = self.backbone.synthesis(ws, coord_embedding, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(ws.shape[0], self.n_planes_per_axis, planes.shape[-3], planes.shape[-2], planes.shape[-1])
        out, sdf, _, _ = self.renderer.run_model(ws[:, -1].clone(), planes, self.decoder, coordinates, directions, real_conditions, self.rendering_kwargs)
        out['sigma'] += sdf.reshape(out['sigma'].shape)
        # out['sigma'] = self.density_func(out['sigma'])
        return out

    def forward(self, z, c, real_conditions, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None,
                update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
                          update_emas=update_emas)
        return self.synthesis(ws, c, real_conditions, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution,
                              cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone,
                              **synthesis_kwargs)


from training.networks_stylegan2 import FullyConnectedLayer


class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'],
                                lr_multiplier=options['decoder_lr_mul'])
        )

    def forward(self, sampled_features, point_embedding, sdf):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features # torch.cat([sampled_features, point_embedding, sdf], dim=-1)

        N, M, C = x.shape
        x = x.view(N * M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = torch.sigmoid(x[..., 1:]) * (1 + 2 * 0.001) - 0.001  # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}


class DeformationDecoder(torch.nn.Module):
    def __init__(self, p_features=3, n_features=512 + 69 + 10, options={'decoder_lr_mul': 1.}):
        super().__init__()
        self.hidden_dim = 128

        self.net_c = torch.nn.Sequential(
            FullyConnectedLayer(p_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.LeakyReLU(.2),
            FullyConnectedLayer(self.hidden_dim, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']))

        self.net_W = [torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.LeakyReLU(.2),
            FullyConnectedLayer(self.hidden_dim, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
        ) for _ in range(3)]

        self.net_b = [torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.LeakyReLU(.2),
            FullyConnectedLayer(self.hidden_dim, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
        ) for _ in range(3)]

        self.net_W = nn.ModuleList(self.net_W)
        self.net_b = nn.ModuleList(self.net_b)
        
        self.decoder = torch.nn.Sequential(
            FullyConnectedLayer(self.hidden_dim, self.hidden_dim // 2, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.LeakyReLU(.2),
            FullyConnectedLayer(self.hidden_dim // 2, self.hidden_dim // 2, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.LeakyReLU(.2),
            FullyConnectedLayer(self.hidden_dim // 2, 3, lr_multiplier=options['decoder_lr_mul']),
        )

    def forward(self, embed_points, latent, real_conditions):
        full_pose = real_conditions['full_pose'][:, 1:, :3, :3].to(embed_points.device)  # [B, 23, 3, 3]
        shapes = real_conditions['body_shape'].to(embed_points.device)  # [B, 10]
        pose_c = torch.zeros([shapes.shape[0], 23, 3], device=embed_points.device)
        pose_c[..., 0] = torch.atan2(full_pose[:, :, 2, 1], full_pose[:, :, 2, 2])
        pose_c[..., 1] = torch.atan2(-full_pose[:, :, 2, 0],
                                     torch.sqrt(full_pose[:, :, 2, 1] ** 2 + full_pose[:, :, 2, 2] ** 2))
        pose_c[..., 2] = torch.atan2(full_pose[:, :, 1, 0], full_pose[:, :, 0, 0])

        B, N, D = embed_points.shape
        embed_points = embed_points.reshape(B * N, D)
        point_feature = self.net_c(embed_points)
        point_feature = point_feature.reshape(B, N, -1)

        W = [net_W(torch.cat([shapes, pose_c.reshape(-1, 69), latent], dim=-1)) for net_W in self.net_W]
        b = [net_b(torch.cat([shapes, pose_c.reshape(-1, 69), latent], dim=-1)) for net_b in self.net_b]

        for i in range(3):
            point_feature = W[i].unsqueeze(1) * point_feature + b[i].unsqueeze(1)
            point_feature = torch.sin(point_feature)

        point_feature = point_feature.reshape(B * N, -1)
        delta_x = self.decoder(point_feature)
        delta_x = torch.sin(delta_x) * 0.3

        delta_x = delta_x.reshape(B, N, 3)
        
        return delta_x
