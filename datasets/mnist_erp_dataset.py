import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from utils.visualization_util import show_spheres, grid_2_points
from utils.download_util import download_mnist
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
                 bandwidth: int = 25,
                 ):
        super().__init__()

        self.root = root
        assert split in ['train', 'test']
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

        if rotate and split == 'test':
            # fix for testing
            np.random.seed(7788)
            self.test_rot_idx = np.random.choice(50000, 10000)

    def __getitem__(self, idx):

        img = self.data[idx]                                  # tensor
        img_np = img.numpy()

        # D-H ERP
        img_np = cv2.resize(img_np, (self.omni_w, self.omni_h))  # dsize of cv2.resize [width, height]
        grid = get_projection_grid(b=self.bandwidth, grid_type='ERP')

        if self.rotate:

            # get random index at training
            if self.split == 'train':
                rot_idx = np.random.randint(0, 50000)
            # get fixed index at testing
            elif self.split == 'test':
                rot_idx = self.test_rot_idx[idx]

            # -------------------------------------- choose one method for remap --------------------------------------
            #  ################### 1) create rotation remap ###################

            # R = rand_rotation_matrix()
            # map_x, map_y = rotate_map_given_R(R, self.omni_h, self.omni_w)
            # img_np = cv2.remap(img_np, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)

            #  ################### 2) load rotation remap ###################
            now_dir = os.getcwd()

            # for dataset test
            map_path_name = 'xy_maps_50000_mnist'  # 'xy_maps_50_50'
            if 'datasets' in now_dir.split('\\'):
                map_matrix_dir = os.path.join(os.path.split(now_dir)[0], map_path_name)
            # for main
            else:
                map_matrix_dir = os.path.join(now_dir, map_path_name)

            map_x_path = map_matrix_dir + '/' + str('%05d' % rot_idx) + '_x.npy'
            map_y_path = map_matrix_dir + '/' + str('%05d' % rot_idx) + '_y.npy'

            map_x = np.load(map_x_path)
            map_y = np.load(map_y_path)
            img_np = cv2.remap(img_np, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)

        # add last channel axis
        img_np = cv2.resize(img_np, (self.bandwidth * 2, self.bandwidth))
        img_np = img_np[:, :, np.newaxis]    # [2 * bandwidth, 2 * bandwidth, 1] [H, W, 1]
        # cv2.imshow('rotated_img', img_np)
        # cv2.waitKey(0)

        points = grid_2_points(grid=grid)  # tuples -> (num_points, 3)

        if self.vis:

            img_np_vis = img_np
            cv2.imshow('rotated_img', img_np_vis)
            cv2.waitKey(0)
            rgb = img_np_vis.reshape(-1, 1)                             # [H * W, 3]
            show_spheres(scale=2, points=points, rgb=rgb)       # points, rgb : (num_points, 3)

        img_np = np.transpose(img_np, (2, 0, 1))
        img_torch = torch.FloatTensor(img_np)                   # [1, 60, 60]

        label = int(self.targets[idx])
        return img_torch, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dataset = Mnist_ERP_Dataset(root='D:\data\MNIST', split='test', rotate=True, vis=True, bandwidth=25)
    img, label = dataset.__getitem__(0)
    print(img.size())