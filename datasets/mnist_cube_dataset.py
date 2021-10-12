import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from samplings.cube_sampling import inflate_cube

from utils.visualization_util import show_spheres
from utils.mnist_download_util import download_mnist
from utils.projection_util import get_projection_grid, cartesian_to_spherical, spherical_to_plane
from utils.rotation_util import calculate_Rmatrix_from_phi_theta, rotate_map_given_R


class Mnist_Cube_Dataset(Dataset):

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
                 num_edge: int = 15,
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
        self.num_edge = num_edge
        self.rotate = rotate
        self.vis = vis
        self.omni_h = self.omni_w = self.bandwidth * 2

        self.cube_face_list = inflate_cube(num_edge=num_edge)
        self.cube_mapping_list = []

        # make mapping matrix for cube to erp
        # loop cube face (6)
        for cube_face in self.cube_face_list:
            num_point = cube_face.shape[0]  # num point, 3
            cube_sampling_map_x = np.zeros(num_point, dtype=np.float32)
            cube_sampling_map_y = np.zeros(num_point, dtype=np.float32)

            # each points convert cartesian(x, y, z) to spherical(phi, theta)
            for n_i in range(num_point):
                [p, t] = cartesian_to_spherical(cube_face[n_i, 0], cube_face[n_i, 1], cube_face[n_i, 2])
                y_on_equi, x_on_equi = spherical_to_plane(p, t, self.omni_h, self.omni_w)
                cube_sampling_map_x[n_i] = x_on_equi
                cube_sampling_map_y[n_i] = y_on_equi
            self.cube_mapping_list.append((cube_sampling_map_x, cube_sampling_map_y))

    def __getitem__(self, idx):

        img = self.data[idx]                                  # tensor
        img_np = img.numpy()

        img_np = cv2.resize(img_np, (self.omni_h, self.omni_w))

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

        equi = img_np
        rotated_equi = equi

        # cube partitioning
        coordinates = []
        patch_list = []
        for coord, maps in zip(self.cube_face_list, self.cube_mapping_list):
            map_x = maps[0]  # (edge ** 2,)
            map_y = maps[1]  # (edge ** 2,)
            bert_input_patch = cv2.remap(rotated_equi, map_x, map_y, cv2.INTER_CUBIC)  # [16, 1]
            coordinates.append(coord)
            patch_list.append(bert_input_patch)

        coordinates = np.array(coordinates)
        patch_list = np.array(patch_list)

        if self.vis:

            print("label : ", int(self.targets[idx]))
            coordinates_vis = coordinates.reshape(6 * self.num_edge ** 2, -1)  # [6 * self.num_edge ** 2, 3]
            cal_vis = patch_list.reshape(6 * self.num_edge ** 2, -1)           # [6 * self.num_edge ** 2, 1]
            show_spheres(scale=2, points=coordinates_vis, rgb=cal_vis)

        sequence_tensor = torch.from_numpy(patch_list).type(torch.float32).squeeze(-1)  # [6, num_edge ^ 2]
        label = int(self.targets[idx])
        return sequence_tensor, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dataset = Mnist_Cube_Dataset(root='D:\data\MNIST', split='test', vis=True, num_edge=15)
    seq, label = dataset.__getitem__(0)
    print(seq.size())