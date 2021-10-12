import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from utils.visualization_util import show_spheres, grid_2_points
from utils.mnist_download_util import download_mnist
from utils.projection_util import get_projection_grid, rotate_grid
from utils.rotation_util import calculate_Rmatrix_from_phi_theta, rotate_map_given_R


class Mnist_ERP_Dataset(Dataset):

    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(self,
                 root: str,
                 split: str,
                 download: bool = False,
                 rotate: bool = True,
                 vis: bool = False,
                 bandwidth: int = 30,
                 ):
        super().__init__()

        self.root = root
        self.split = split  # training set or test set

        if download:
            download_mnist(root)

        if self.split == 'train':
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(os.path.join(root, 'MNIST', 'processed'), data_file))

        self.bandwidth = bandwidth
        self.rotate = rotate
        self.vis = vis
        self.omni_h = self.omni_w = self.bandwidth * 2

    def __getitem__(self, idx):

        img = self.data[idx]                                  # tensor
        img_np = img.numpy()

        # D-H ERP
        img_np = cv2.resize(img_np, (self.omni_h, self.omni_w))
        grid = get_projection_grid(b=self.bandwidth)

        if self.rotate:

            # get random rotation (s2 - phi, theta)
            phi = np.random.randint(0, 180)
            theta = np.random.randint(0, 180) * 2
            # phi = theta = 0
            # print(phi, theta)

            # -------------------------------------- choose one method for remap --------------------------------------

            # 1) create rotation remap

            # R = calculate_Rmatrix_from_phi_theta(phi, theta)
            # map_x, map_y = rotate_map_given_R(R, self.omni_h, self.omni_w)
            # img_np = cv2.remap(img_np, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)

            # 2) load rotation remap

            map_matrix_dir = r'C:\\Users\csm81\Desktop\projects_4 (transformer)\\new_20210901_360\\360bert\xy_maps'
            map_x_path = map_matrix_dir + '/' + str('%03d' % phi) + '_' + str('%03d' % theta) + '_x.npy'
            map_y_path = map_matrix_dir + '/' + str('%03d' % phi) + '_' + str('%03d' % theta) + '_y.npy'
            map_x = np.load(map_x_path)
            map_y = np.load(map_y_path)

            img_np = cv2.remap(img_np, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)

        # add last channel axis
        img_np = img_np[:, :, np.newaxis]    # [2 * bandwidth, 2 * bandwidth, 1] [H, W, 1]

        if self.vis:
            if self.rotate:
                rot = calculate_Rmatrix_from_phi_theta(phi, theta)
                rotated_grid = rotate_grid(rot, grid)
            else:
                rotated_grid = grid

            img_np_vis = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)       # RGB - [H, W, 3]
            # cv2.imshow('rotated_img', img_np_vis)
            # cv2.waitKey(0)

            rgb = img_np_vis.reshape(-1, 3)                             # [H * W, 3]
            rotated_points = grid_2_points(grid=rotated_grid)           # tuples -> (num_points, 3)
            show_spheres(scale=2, points=rotated_points, rgb=rgb)       # points, rgb : (num_points, 3)

        img_np = np.transpose(img_np, (2, 0, 1))
        img_torch = torch.FloatTensor(img_np)                   # [1, 60, 60]

        label = int(self.targets[idx])
        return img_torch, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dataset = Mnist_ERP_Dataset(root='D:\data\MNIST', split='test', vis=True, bandwidth=26)
    img, label = dataset.__getitem__(0)
    print(img.size())