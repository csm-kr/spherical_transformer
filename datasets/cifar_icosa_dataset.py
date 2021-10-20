import os
import cv2
import torch
import numpy as np
import pickle
from typing import Any, Callable, Optional, Tuple

from torch.utils.data import Dataset, DataLoader

from samplings.icosahedron_sampling import inflate_icosahedron
from utils.visualization_util import show_spheres, grid_2_points
from utils.download_util import download_cifar10, check_integrity
from utils.projection_util import get_projection_grid, rotate_grid, cartesian_to_spherical, spherical_to_plane
from utils.rotation_util import calculate_Rmatrix_from_phi_theta, rotate_map_given_R


class Cifar_Icosa_Dataset(Dataset):


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
                 bandwidth: int = 50,
                 division_level: int = 4
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
        self.division_level = division_level
        self.rotate = rotate
        self.vis = vis
        self.omni_h = self.omni_w = self.bandwidth * 2

        self.phi_fov = 65.5
        self.theta_fov = 65.5

        if rotate and split == 'test':
            # fix for testing
            np.random.seed(7788)
            self.test_rot_idx = np.random.choice(50000, 10000)

        self.icosa_face_list = inflate_icosahedron(division_level=division_level)
        self.icosa_mapping_list = []

        # make mapping matrix for icosahedron to erp
        # loop cube face (20)
        for icosa_face in self.icosa_face_list:
            num_point = icosa_face.shape[0]  # num point, 3
            icosa_sampling_map_x = np.zeros(num_point, dtype=np.float32)
            icosa_sampling_map_y = np.zeros(num_point, dtype=np.float32)

            # each points convert cartesian(x, y, z) to spherical(phi, theta)
            for n_i in range(num_point):
                [p, t] = cartesian_to_spherical(icosa_face[n_i, 0], icosa_face[n_i, 1], icosa_face[n_i, 2])
                y_on_equi, x_on_equi = spherical_to_plane(p, t, self.omni_h, self.omni_w)
                icosa_sampling_map_x[n_i] = x_on_equi
                icosa_sampling_map_y[n_i] = y_on_equi
            self.icosa_mapping_list.append((icosa_sampling_map_x, icosa_sampling_map_y))

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
        img_np = img
        img_np = cv2.resize(img_np, (self.omni_w, self.omni_h))  # dsize of cv2.resize [width, height]
        # cv2.imshow('input', img_np)
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

        equi = img_np
        rotated_equi = equi

        # icosa partitioning
        coordinates = []
        patch_list = []
        for coord, maps in zip(self.icosa_face_list, self.icosa_mapping_list):
            map_x = maps[0]  # (edge ** 2,)
            map_y = maps[1]  # (edge ** 2,)
            bert_input_patch = cv2.remap(rotated_equi, map_x, map_y, cv2.INTER_CUBIC)  # [16, 1]
            coordinates.append(coord)
            patch_list.append(bert_input_patch)

        coordinates = np.array(coordinates)
        patch_list = np.array(patch_list)

        if self.vis:
            print("label : ", int(self.targets[idx]))
            coordinates_vis = coordinates.reshape(20 * 4 ** self.division_level,
                                                  -1)  # [20 * 4 ** self.division_level, 3]
            cal_vis = patch_list.reshape(20 * 4 ** self.division_level, -1)  # [20 * 4 ** self.division_level, 1]
            show_spheres(scale=2, points=coordinates_vis, rgb=cal_vis)

        sequence_tensor = torch.from_numpy(patch_list).type(torch.float32).squeeze(-1)  # [20, 4 ** self.division_level]
        sequence_tensor = sequence_tensor.reshape(20, -1)
        label = int(self.targets[idx])
        return sequence_tensor, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dataset = Cifar_Icosa_Dataset(root='D:\data\CIFAR10', split='test', rotate=True, vis=True, bandwidth=50, division_level=4)
    img, label = dataset.__getitem__(0)
    print(img.size())