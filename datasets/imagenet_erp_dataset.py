import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.imagenet import load_meta_file
from torchvision.datasets.folder import ImageFolder

from utils.visualization_util import show_spheres, grid_2_points
from utils.mnist_download_util import download_mnist
from utils.projection_util import get_projection_grid, rotate_grid, make_fov_projection_map
from utils.rotation_util import calculate_Rmatrix_from_phi_theta, rotate_map_given_R


class ImageNet_ERP_Dataset(ImageFolder):

    def __init__(self,
                 root: str,
                 split: str,
                 download: bool = False,
                 rotate: bool = True,
                 vis: bool = False,
                 bandwidth: int = 112,
                 ):

        self.root = root
        assert split in ('train', 'val')
        self.split = split  # training set or test set
        self.rotate = rotate
        self.vis = vis

        wnid_to_classes = load_meta_file(self.root)[0]

        super(ImageNet_ERP_Dataset, self).__init__(self.split_folder)
        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx
                             for idx, clss in enumerate(self.classes)
                             for cls in clss}

        self.bandwidth = bandwidth
        self.omni_h = self.omni_w = self.bandwidth * 2

        self.phi_fov = 65.5
        self.theta_fov = 65.5


    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, self.split)

    def __getitem__(self, idx):

        path, target = self.samples[idx]
        img = self.loader(path)
        img_np = np.array(img)
        img_np = cv2.resize(img_np, (self.omni_h, self.omni_w))

        # D-H ERP
        fov_proj_map_x, fov_proj_map_y, = make_fov_projection_map(self.omni_h, self.omni_w, self.phi_fov, self.theta_fov)
        undefined_zone = fov_proj_map_y == -1
        OMNI_H, OMNI_W = fov_proj_map_x.shape
        img_np = cv2.remap(img_np, fov_proj_map_x, fov_proj_map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
        img_np[undefined_zone] = 0

        img_np = cv2.resize(img_np, (self.omni_h, self.omni_w))
        grid = get_projection_grid(b=self.bandwidth)

        if self.rotate:
            # get random rotation (s2 - phi, theta)
            phi = np.random.randint(0, 180)
            theta = np.random.randint(0, 180) * 2
            # phi = theta = 0
            print(phi, theta)

            # from scratch
            R = calculate_Rmatrix_from_phi_theta(phi, theta)
            map_x, map_y = rotate_map_given_R(R, self.omni_h, self.omni_w)
            img_np = cv2.remap(img_np, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
            # cv2.imshow('rotated_img', img_np)
            # cv2.waitKey(0)

            # map_matrix_dir = '../xy_maps/'
            # map_matrix_dir = r'C:\\Users\csm81\Desktop\projects_4 (transformer)\\new_20210901_360\\360bert\xy_maps'
            # map_x_path = map_matrix_dir + '/' + str('%03d' % phi) + '_' + str('%03d' % theta) + '_x.npy'
            # map_y_path = map_matrix_dir + '/' + str('%03d' % phi) + '_' + str('%03d' % theta) + '_y.npy'
            # map_x = np.load(map_x_path)
            # map_y = np.load(map_y_path)
            # img_np = cv2.remap(img_np, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)

        # add last channel axis
        # img_np = img_np[:, :, np.newaxis]  # [2 * bandwidth, 2 * bandwidth, 1] [H, W, 1]

        if self.vis:
            if self.rotate:
                rot = calculate_Rmatrix_from_phi_theta(phi, theta)
                rotated_grid = rotate_grid(rot, grid)
            else:
                rotated_grid = grid

            img_np_vis = img_np
            # cv2.imshow('rotated_img', img_np_vis)
            # cv2.waitKey(0)
            rgb = img_np_vis.reshape(-1, 3)  # [H * W, 3]
            rotated_points = grid_2_points(grid=rotated_grid)  # tuples -> (num_points, 3)
            show_spheres(scale=2, points=rotated_points, rgb=rgb)  # points, rgb : (num_points, 3)

        img_np = np.transpose(img_np, (2, 0, 1))  # [3, 224, 224]
        img_torch = torch.FloatTensor(img_np)  # [3, 224, 224]

        label = int(self.targets[idx])
        return img_torch, label

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    dataset = ImageNet_ERP_Dataset(root='D:\data\ILSVRC_classification', split='train', vis=True)
    dataset.__getitem__(0)