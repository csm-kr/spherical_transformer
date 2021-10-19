import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from samplings.icosahedron_sampling import inflate_icosahedron

from utils.visualization_util import show_spheres
from utils.download_util import download_mnist
from utils.projection_util import get_projection_grid, cartesian_to_spherical, spherical_to_plane
from utils.rotation_util import calculate_Rmatrix_from_phi_theta, rotate_map_given_R, rand_rotation_matrix


class Mnist_Icosa_Dataset(Dataset):

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
                 division_level: int = 3,
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
        self.division_level = division_level
        self.rotate = rotate
        self.vis = vis
        self.omni_h = self.omni_w = self.bandwidth * 2

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

    def __getitem__(self, idx):

        img = self.data[idx]                                  # tensor
        img_np = img.numpy()

        # D-H ERP
        img_np = cv2.resize(img_np, (self.omni_w, self.omni_h))

        if self.rotate:

            # get random index at training
            if self.split == 'train':
                rot_idx = np.random.randint(0, 50000)
            # get fixed index at testing
            elif self.split == 'test':
                rot_idx = self.test_rot_idx[idx]

            # phi = theta = 0
            # print(phi, theta)

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

        equi = img_np
        rotated_equi = equi

        # cube partitioning
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
            coordinates_vis = coordinates.reshape(20 * 4 ** self.division_level, -1)  # [20 * 4 ** self.division_level, 3]
            cal_vis = patch_list.reshape(20 * 4 ** self.division_level, -1)           # [20 * 4 ** self.division_level, 1]
            show_spheres(scale=2, points=coordinates_vis, rgb=cal_vis)

        sequence_tensor = torch.from_numpy(patch_list).type(torch.float32).squeeze(-1)  # [20, 4 ** self.division_level]
        label = int(self.targets[idx])
        return sequence_tensor, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dataset = Mnist_Icosa_Dataset(root='D:\data\MNIST', split='test', rotate=True, vis=True, division_level=3)
    seq, label = dataset.__getitem__(0)
    print(seq.size())