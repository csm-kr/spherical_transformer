import os
import cv2
import torch
import numpy as np
import pickle
from typing import Any, Callable, Optional, Tuple

from torch.utils.data import Dataset, DataLoader

from utils.visualization_util import show_spheres, grid_2_points
from utils.download_util import download_cifar10, check_integrity
from utils.projection_util import get_projection_grid, rotate_grid
from utils.rotation_util import calculate_Rmatrix_from_phi_theta, rotate_map_given_R


class Cifar_ERP_Dataset(Dataset):


    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }
    base_folder = 'cifar-10-batches-py'

    def __init__(self,
                 root: str,
                 split: str,
                 download: bool = True,
                 rotate: bool = True,
                 vis: bool = False,
                 bandwidth: int = 25,
                 ):
        super().__init__()

        self.root = root

        assert split in ['train', 'test']
        self.split = split  # training set or test set

        if download:
            download_cifar10(root)

        if self.split == 'train':
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self._load_meta()
        # load self.data and self.targets

        self.bandwidth = bandwidth
        self.rotate = rotate
        self.vis = vis
        self.omni_h = self.omni_w = self.bandwidth * 2

        self.phi_fov = 65.5
        self.theta_fov = 65.5

        if rotate and split == 'test':
            # fix for testing
            np.random.seed(7788)
            self.test_rot_idx = np.random.choice(50000, 10000)

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def __getitem__(self, idx):

        img = self.data[idx]                                  # tensor
        # print(self.targets[idx])
        img_np = img
        img_np = cv2.resize(img_np, (self.omni_w, self.omni_h))  # dsize of cv2.resize [width, height]
        # cv2.imshow('rotated_img', img_np)
        # cv2.waitKey(0)

        # -------------------------------- load fov proj --------------------------------
        now_dir = os.getcwd()
        map_path_name = 'xy_fov_maps_100'
        if 'datasets' in now_dir.split('\\'):
            map_matrix_dir = os.path.join(os.path.split(now_dir)[0], map_path_name)
        else:
            map_matrix_dir = os.path.join(now_dir, map_path_name)
        map_x_path = map_matrix_dir + '/' + 'x.npy'
        map_y_path = map_matrix_dir + '/' + 'y.npy'
        fov_proj_map_x = np.load(map_x_path)
        fov_proj_map_y = np.load(map_y_path)
        undefined_zone = fov_proj_map_y == -1
        img_np = cv2.remap(img_np, fov_proj_map_x, fov_proj_map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
        img_np[undefined_zone] = 0

        img_np = cv2.resize(img_np, (self.omni_h, self.omni_w))
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
            map_path_name = 'xy_maps_50000_cifar'  # 'xy_maps_50_50'
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
        points = grid_2_points(grid=grid)  # tuples -> (num_points, 3)

        if self.vis:

            img_np_vis = img_np
            # cv2.imshow('rotated_img', img_np_vis)
            # cv2.waitKey(0)
            rgb = img_np_vis.reshape(-1, 3)                             # [H * W, 3]
            show_spheres(scale=2, points=points, rgb=rgb)       # points, rgb : (num_points, 3)

        img_np = np.transpose(img_np, (2, 0, 1))
        img_torch = torch.FloatTensor(img_np)                   # [1, 60, 60]

        label = int(self.targets[idx])
        return img_torch, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dataset = Cifar_ERP_Dataset(root='D:\data\CIFAR10', split='test', rotate=True, vis=True, bandwidth=50)
    img, label = dataset.__getitem__(0)
    print(img.size())