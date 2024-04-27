import sys, os

sys.path.insert(0, os.getcwd())

import cv2
import argparse
import numpy as np
import os
import torch
from tqdm import tqdm
import configs
import imageio
import json
from torch.utils.data import Dataset, DataLoader
import joblib
from pytorch3d.transforms import euler_angles_to_matrix
from scipy.spatial.transform import Rotation
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
torch.cuda.manual_seed(0)
import pickle

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--smpl_folder', type=str)
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--n_poses', type=int, default=None)
    opt = parser.parse_args()

    return opt


if __name__ == '__main__':

    opt = parse_args()
    os.makedirs(opt.output_dir, exist_ok=True)
    
    all_poses = os.listdir(opt.smpl_folder)
    all_poses.sort()
    if opt.n_poses is None:
        opt.n_poses = len(all_poses)
        print('Process all poses in the folder')
    elif opt.n_poses > len(all_poses):
        opt.n_poses = len(all_poses)
        print('Only {} poses in the folder, process all poses in the folder'.format(len(dataset)))
    else:
        print('Process {all} poses in the folder'.format(opt.n_poses))
        
    for i in tqdm(range(opt.n_images)):
        smpl = joblib.load(os.path.join(opt.smpl_folder, all_poses[i])) 
        
        fov = np.pi * 12 / 180
        focal = 1. / np.tan(fov / 2)
        sx, sy, tx, ty = smpl['orig_cam'][0].astype(np.float32)
        sx = sx / 2.

        camera_K = np.array([
            [focal, 0, 0, 0],
            [0, focal, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        camera_R = np.eye(4)
        camera_T = np.array([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, focal / sx],
            [0, 0, 0, 1],
        ])        

        joints = [i for i in range(24)]
         
        # skeleton
        skeleton_xyz = smpl['joints'][0].astype(np.float32)
        skeleton_xyz = skeleton_xyz[joints]
        
        # global orient
        global_orient = smpl["global_orient"][0].astype(np.float32)

        # body pose \theta
        body_pose = smpl['full_pose'][0]

        # t pose with shape condition
        tpose_vertices_shaped = smpl['tpose_vertices'][0]

        # fk matrices
        fk_matrices = smpl['fk_matrices'][0]

        # inv of root rotation
        inverse_root = np.linalg.inv(body_pose[0])
        cano_rotation = Rotation.from_euler('xyz', [np.pi, 0, 0]).as_matrix()
        cano_matrix = np.eye(4)
        cano_matrix[:3, :3] = cano_rotation @ inverse_root
        fk_matrices = np.einsum("ij,bjk->bik", cano_matrix, fk_matrices)

        # linear blending skinning
        lbs_weights = smpl['lbs_weights']
        vertice_fk_matrices = np.einsum("bi,ijk->bjk", lbs_weights, fk_matrices)
        tpose_vertices_homo = np.pad(tpose_vertices_shaped, [[0, 0], [0, 1]], "constant", constant_values=1.)
        # vertices with pose
        vertices = np.einsum("bij,bj->bi", vertice_fk_matrices, tpose_vertices_homo)[:, :3]

        # update skeleton xyz
        skeleton_homo = np.pad(skeleton_xyz, [[0, 0], [0, 1]], "constant", constant_values=1.)
        skeleton_xyz = np.einsum('ij,bj->bi', cano_matrix, skeleton_homo)[:, :3]

        tpose_vertices = tpose_vertices_shaped
        tpose_vertices[..., 1] += .35
        
        ret_dict = {
            'fk_matrices': fk_matrices,
            'vertices': vertices,
            'lbs_weights': lbs_weights
        }
        
        pickle.dump(
            ret_dict, open(os.path.join(opt.output_folder, all_poses[i].replace('.pkl', '.pickle')), 'wb')
        )
