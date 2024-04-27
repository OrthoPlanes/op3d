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

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
torch.cuda.manual_seed(0)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str)
    parser.add_argument('--smpl_folder', type=str)
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--n_images', type=int, default=None)
    parser.add_argument('--focal_length', type=float, default=2985.29/700)
    parser.add_argument('--camera_radius', type=float, default=2.7)
    parser.add_argument('--tune', type=str, default='')
    parser.add_argument('--variant', type=int, default=0)
    opt = parser.parse_args()

    return opt


class TempDataset():
    def __init__(self, image_folder, smpl_folder):
        self.image_folder = image_folder
        self.smpl_folder = smpl_folder
        
        self.all_iamges = os.listdir(self.image_folder)
        self.all_poses = os.listdir(self.smpl_folder)
        self.all_iamges.sort()
        self.all_poses.sort()
        assert len(self.all_iamges) == len(self.all_poses), 'Some images are not paired with poses...'
        
    def __len__(self):
        return len(self.all_iamges)
        
    def __getitem__(self, index):
        image_file_name = self.all_iamges[index]
        pose_file_name = image_file_name.replace(image_file_name.split('.')[-1], 'pkl')
        
        image = cv2.imread(os.path.join(self.image_folder, image_file_name))[..., [2, 1, 0]]
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.
        image = image.float()
        
        pose = joblib.load(os.path.join(self.smpl_folder, pose_file_name)) 
        human_pose = pose[list(pose.keys())[0]]['pose'].reshape(1, 72).reshape(24, 3)
        
        global_orient_theta = human_pose[0]
        R = euler_angles_to_matrix(torch.as_tensor(global_orient_theta).reshape(1, 3), convention="XYZ")
        
        return {
            'image': image,
            'global_orient': R
        }

if __name__ == '__main__':

    opt = parse_args()
    os.makedirs(opt.output_dir, exist_ok=True)
    
    dataset_beforeprocess = TempDataset()
    dataiter = iter(dataset_beforeprocess)

    if opt.n_images is None:
        opt.n_images = len(dataset_beforeprocess)
        print('Process all images in the folder')
    elif opt.n_images > len(dataset_beforeprocess):
        opt.n_images = len(dataset_beforeprocess)
        print('Only {} images in the folder, process all images in the folder'.format(len(dataset_beforeprocess)))
    else:
        print('Process {all} images in the folder'.format(opt.n_images))

    dataset = {'labels': []}

    for i in tqdm(range(opt.n_images)):

        data = next(dataiter)

        out_path = os.path.join(opt.output_dir, f"{i:06d}.png")

        if not os.path.exists(out_path):
            image = data["image"][0].permute([1, 2, 0]).numpy()
            image = np.round((image + 1.) * 0.5 * 255).clip(0, 255).astype(np.uint8)
            imageio.imwrite(out_path, image)

        intrinsics = np.array([
            [opt.focal_length, 0, 0.5],
            [0, opt.focal_length, 0.5],
            [0, 0, 1]
        ])

        R = np.eye(4)
        R[:3, :3] = data["global_orient"][0].cpu().numpy()
        T = np.eye(4)
        T[2, 3] = opt.camera_radius

        cam2world_matrix = np.linalg.inv(T @ R)

        label = np.concatenate([cam2world_matrix.reshape(-1), intrinsics.reshape(-1)]).tolist()
        dataset["labels"].append([f"{i:06d}.png", label])

    with open(os.path.join(opt.output_dir, "dataset.json"), "w") as f:
        json.dump(dataset, f, indent=4)
        