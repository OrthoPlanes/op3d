# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])
        # load rebalance data

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
            self._raw_labels_std = self._raw_labels.std(0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    def get_label_std(self):
        return self._raw_labels_std

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels
'''
#----------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Ceph
import ctypes
import fnmatch
import importlib
import inspect
import numpy as np
import os
import shutil
import sys
import types
import io
import pickle
import re
import requests
import html
import hashlib
import glob
import tempfile
import urllib
import urllib.request
import uuid
import argparse
import pickle
import time

from distutils.util import strtobool
from typing import Any, List, Tuple, Union
from petrel_client.client import Client
import cv2
from PIL import Image

# Util classes
# ------------------------------------------------------------------------------------------

class Ceph:

    def __init__(self, bucket, config="petreloss_s_cluster.conf"):

        self.bucket = bucket
        self.client = Client(config)

    def get_bytes(self, path):
        for i in range(10):
            bytes = self.client.get(path)
            if bytes is not None: break
            time.sleep(0.1)
        return bytes

    def get_image(self, path):

        if not path.startswith("s3://"): path = os.path.join(self.bucket, path)
        bytes = self.get_bytes(path)
        if bytes is None: print("error loading {}".format(path))
        img_mem_view = memoryview(bytes)
        img_array = np.frombuffer(img_mem_view, np.uint8)
        img_array = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = Image.fromarray(img_array)

        return img

    def get_numpy_txt(self, path):

        if not path.startswith("s3://"): path = os.path.join(self.bucket, path)
        bytes = self.get_bytes(path)
        if bytes is None: print("error loading {}".format(path))
        return np.loadtxt(io.BytesIO(bytes))

    def get_pkl(self, path, python2=False):
        if not path.startswith("s3://"): path = os.path.join(self.bucket, path)
        bytes = self.get_bytes(path)
        if bytes is None: print("error loading {}".format(path))
        if python2:
            ret = pickle.loads(bytes, encoding="latin1")
        else:
            ret = pickle.loads(bytes)
        return ret

    def put_image(self, path, data):
        if not path.startswith("s3://"): path = os.path.join(self.bucket, path)
        _, ext = os.path.splitext(path)
        success, array = cv2.imencode("." + ext, data)
        return self.client.put(path, array.tostring())

    def put_pkl(self, path, data):
        if not path.startswith("s3://"): path = os.path.join(self.bucket, path)
        bytes = pickle.dumps(data)
        return self.client.put(path, bytes)


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


class Logger(object):
    """Redirect stderr to stdout, optionally print stdout to a file, and optionally force flushing on both stdout and the file."""

    def __init__(self, file_name: str = None, file_mode: str = "w", should_flush: bool = True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def write(self, text: Union[str, bytes]) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if isinstance(text, bytes):
            text = text.decode()
        if len(text) == 0: # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()
            self.file = None


def print_stats(input, name="input", scientific=False):
    dims = input.shape[-1]
    template = "{} dim={}: min={:.3e}, mean={:.3e}, max={:.3e}, std={:.3e}" if scientific else "{} dim={}: min={:.3f}, mean={:.3f}, max={:.3f}, std={:.3f}"
    for i in range(dims):
        x = input[..., i]
        print(template.format(
            name, i, x.min(), x.mean(), x.max(), x.std()
        ))

import joblib
from torchvision import transforms
from scipy.spatial.transform import Rotation


class CephSHHQDataset(torch.utils.data.Dataset):
    def __init__(self,
        path,
        resolution  = None,
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
        down        = 1,
    ):

        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # TODO: W/O Xflip

        # root
        self.root = 's3://fbhqv2'

        self.length = 219047
        self._index = np.arange(self.length, dtype=np.int64)
        # Apply max_size.
        if (max_size is not None) and (self.length > max_size):
            np.random.RandomState(random_seed).shuffle(self._index)
            self._index = np.sort(self._index[:max_size])
            self.length = max_size

        self.corrupted = [118464]
        self.down = down
        self.render_resolution = 128
        ceph_config = '~/petreloss_s_cluster.conf'
        self.ceph = Ceph(self.root, config=ceph_config)
        print("SHHQ Dataset Init [Using ceph {}]".format(ceph_config))

        self.vertex_downsample = 1
        self.smpl_initial_tpose_vertices, self.smpl_initial_faces, self.geodist_prior, self.vertices_approximation = \
            self.initial_smpl(gaussian_sigma=0.01)

        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            transforms.Resize((512 // self.down, 256 // self.down), interpolation=Image.BILINEAR)])

        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            pass
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)

    def initial_smpl(self, gaussian_sigma=.1):
        print('===========SHHQDataset: INIT SMPL===========')

        # pkl path
        smpl_path = os.path.join(self.root, "SMPL_NEUTRAL.pkl")
        geodists_path = os.path.join(self.root, "geodists_smpl_6890.pkl")
        geodists_cache_path = os.path.join("./", "geodists_smpl_6890.pkl")

        # load smpl
        if True:
            smpl = self.ceph.get_pkl(smpl_path, python2=True)
        else:
            with open(smpl_path, "rb") as f:
                pickle.load(f, encoding='latin1')

        # load geodists
        if os.path.exists(geodists_cache_path):
            smpl_geodists = joblib.load(geodists_cache_path)
        elif True:
            smpl_geodists = self.ceph.get_pkl(geodists_path)
            joblib.dump(smpl_geodists, geodists_cache_path)
        else:
            smpl_geodists = joblib.load(geodists_path)

        # inital T
        smpl_initial_t_pose = smpl['v_template']
        # initial face
        smpl_faces = smpl['f']

        # geodists
        smpl_geodists = smpl_geodists[:, ::self.vertex_downsample]  # [6890, downsampled]
        vertex_approximation = np.argmin(smpl_geodists, axis=1)  # [6890]
        smpl_geodists = smpl_geodists[::self.vertex_downsample]  # [downsampled, downsampled]

        smpl_geodists = torch.from_numpy(smpl_geodists)
        geodist_prior = torch.softmax(-smpl_geodists / gaussian_sigma, dim=1)
        geodist_prior = torch.nn.functional.pad(geodist_prior, (2, 0, 2, 0), mode="constant", value=0.)
        geodist_prior[0, 0] = geodist_prior[1, 1] = 1
        geodist_prior = geodist_prior.numpy()

        return smpl_initial_t_pose, smpl_faces, geodist_prior, vertex_approximation

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self.length
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
            self._raw_labels_std = self._raw_labels.std(0)
        return self._raw_labels

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        rgb_path = os.path.join(self.root, "images", f"{raw_idx + 1:06d}.png")
        rgb = np.array(self.ceph.get_image(rgb_path))
        # rgb = self.image_transform(rgb)

        mask_path = os.path.join(self.root, "masks", f"{raw_idx + 1:06d}.png")
        mask = np.array(self.ceph.get_image(mask_path))
        rgb[mask == 0] = 255

        rgb = self.image_transform(rgb)

        image = rgb.reshape(3, 512 // self.down, 256 // self.down).permute([1, 2, 0]).numpy()
        image = np.pad(image, ((0, 0), (128 // self.down, 128 // self.down), (0, 0)), "constant", constant_values=1.)
        image = np.round((image + 1.) * 0.5 * 255).clip(0, 255).astype(np.uint8)

        image = image.transpose(2, 0, 1)  # HWC => CHW

        return image

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None, zipfile=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        index = self._index[idx]
        while index in self.corrupted:
            index = (index + 1) % 219047

        image = self._load_raw_image(index)

        assert isinstance(image, np.ndarray)
        assert image.dtype == np.uint8

        return image.copy(), self.get_label(idx), self.get_conditions(idx)

    def get_label(self, idx):
        index = self._index[idx]
        # while index in self.corrupted:
        #     index = self._index[(idx + 1) % self.length]

        label = self._get_raw_labels()[index]

        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_conditions(self, idx):
        index = self._index[idx]
        while index in self.corrupted:
            index = (index + 1) % 219047

        smpl_path = os.path.join(self.root, "smpl", f"{index + 1:06d}.pkl")
        smpl = self.ceph.get_pkl(smpl_path)

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

        # # TODO: Canonical Rotation Matrix
        cano_rotation = Rotation.from_euler('xyz', [np.pi, 0, 0]).as_matrix()
        cano_matrix = np.eye(4)
        cano_matrix[:3, :3] = cano_rotation @ inverse_root
        # fk_matrices_cp = fk_matrices
        # cano_matrix_ro = np.eye(4)
        # cano_matrix_ro[:3, :3] = inverse_root
        # fk_matrices_cp = np.einsum("ij,bjk->bik", cano_matrix_ro, fk_matrices)
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

        # normal
        original_normal_data = cv2.imread(f'/mnt/lustre/hehonglin/ECON/results/econ/png/{index + 1:06d}_normal_F.png')
        original_smpl_normal_data = cv2.imread(f'/mnt/lustre/hehonglin/ECON/results/econ/png/{index + 1:06d}_smpl_F.png')

        normal = np.pad(original_normal_data, ((0, 0), (256, 256), (0, 0)), "constant", constant_values=1.)
        normal = normal.clip(0, 255).astype(np.uint8)
        normal = cv2.resize(normal, (self.render_resolution, self.render_resolution))
        normal = normal.transpose(2, 0, 1)[[2, 1, 0], :, :]
        normal = normal / 255
        normal = (2 * normal) - 1

        smpl_normal = np.pad(original_smpl_normal_data, ((0, 0), (256, 256), (0, 0)), "constant", constant_values=1.)
        smpl_normal = smpl_normal.clip(0, 255).astype(np.uint8)
        smpl_normal = cv2.resize(smpl_normal, (self.render_resolution, self.render_resolution))
        smpl_normal = smpl_normal.transpose(2, 0, 1)[[2, 1, 0], :, :]
        smpl_normal = smpl_normal / 255
        smpl_normal = (2 * smpl_normal) - 1

        conditions = {
            # Camera Related Parameters
            # "cano_matrices": cano_matrix.astype(np.float32),
            # "scales": sx.astype(np.float32),
            # "intrinsics": camera_K.astype(np.float32),
            # "R": camera_R.astype(np.float32),
            # "T": camera_T.astype(np.float32),
            # Human Info
            "full_pose": body_pose.astype(np.float32),
            "fk_matrices": fk_matrices.astype(np.float32),
            "lbs_weights": lbs_weights.astype(np.float32),
            # "skeletons_xyz": skeleton_xyz.astype(np.float32),
            "body_shape": smpl["betas"][0].astype(np.float32),
            "vertices": vertices.astype(np.float32),
            "tpose_vertices": tpose_vertices.astype(np.float32),
            "normal": normal,
            "smpl_normal": smpl_normal,
            "global_orient": global_orient,
            "name": f'{index + 1:06d}',
        }

        return conditions

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

    def get_details(self, idx):
        index = self._index[idx]
        # while index in self.corrupted:
        #     index = self._index[(idx + 1) % self.length]

        d = dnnlib.EasyDict()
        d.raw_idx = int(index)
        # d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    def get_label_std(self):
        return self._raw_labels_std

    @property
    def name(self):
        return 'SHHQ'

    @property
    def image_shape(self):
        return [3, 512 // self.down, 512 // self.down]

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):

        return 512 // self.down

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64
'''
