import os
import cv2
import torch
import numpy as np
import glob

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.imagenet import load_meta_file
from torchvision.datasets.folder import ImageFolder


from samplings.cube_sampling import inflate_cube
from utils.visualization_util import show_spheres, grid_2_points
from utils.projection_util import get_projection_grid, rotate_grid, cartesian_to_spherical, spherical_to_plane, make_fov_projection_map
from utils.rotation_util import calculate_Rmatrix_from_phi_theta, rotate_map_given_R


class Panoramic_Cube_Dataset(Dataset):

    class_names = ('bathroom', 'beach', 'bedroom', 'cave', 'forest',
                   'mountain', 'ruin', 'swimming_pool', 'theater', 'train')
    class_names = sorted(class_names)

    def __init__(self,
                 root: str,
                 split: str,
                 download: bool = False,
                 rotate: bool = True,
                 vis: bool = False,
                 bandwidth: int = 100,
                 num_edge: int = 58
                 ):

        super().__init__()

        self.root = root
        assert split in ('train', 'test')
        self.split = split  # training set or test set
        self.rotate = rotate
        self.vis = vis

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}
        self.class_dict_inv = {i: class_name for i, class_name in enumerate(self.class_names)}

        self.bandwidth = bandwidth
        self.num_edge = num_edge
        self.rotate = rotate
        self.vis = vis
        self.omni_h = self.omni_w = self.bandwidth * 2

        train_path = []
        test_path = []
        for class_name in self.class_names:
            img_list = glob.glob(os.path.join(root, class_name) + '/*.jpg')

            num_train_data = int(0.8 * len(img_list))

            np.random.seed(1)
            train_indices = sorted(np.random.choice(len(img_list), num_train_data, replace=False))
            test_indices = sorted(list(set(np.arange(len(img_list))) - set(train_indices)))

            for train_index in train_indices:
                train_path.append(img_list[train_index])

            for test_index in test_indices:
                test_path.append(img_list[test_index])

        if split == 'train':
            self.img_path = train_path
        else:
            self.img_path = test_path

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
                y_on_equi, x_on_equi = spherical_to_plane(p, t, self.omni_h * 3, self.omni_w * 3)
                cube_sampling_map_x[n_i] = x_on_equi
                cube_sampling_map_y[n_i] = y_on_equi
            self.cube_mapping_list.append((cube_sampling_map_x, cube_sampling_map_y))


    def __getitem__(self, idx):

        img = cv2.imread(self.img_path[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        class_name = os.path.dirname(self.img_path[idx]).split('\\')[-1]
        label = self.class_dict[class_name]
        img_np = np.array(img)
        img_np = cv2.resize(img_np, (self.omni_h * 3, self.omni_w * 3))

        if self.rotate:

            rot_idx = np.random.randint(0, 50000)
            now_dir = os.getcwd()

            # for dataset test
            map_path_name = r'D:\data\\xy_maps_50000_image_600'  # 'xy_maps_50_50'
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
            print("label : ", int(label))
            coordinates_vis = coordinates.reshape(6 * self.num_edge ** 2, -1)  # [6 * self.num_edge ** 2, 3]
            cal_vis = patch_list.reshape(6 * self.num_edge ** 2, -1)  # [6 * self.num_edge ** 2, 3]
            show_spheres(scale=2, points=coordinates_vis, rgb=cal_vis)

        sequence_tensor = torch.from_numpy(patch_list).type(torch.float32).squeeze(-1)  # [6, num_edge ^ 2]
        sequence_tensor = sequence_tensor.reshape(6, -1)
        return sequence_tensor, label

    def __len__(self):
        return len(self.img_path)


if __name__ == '__main__':
    dataset = Panoramic_Cube_Dataset(root='D:\data\panorama_360', split='train', rotate=True, vis=True)
    print(len(dataset))
    for data in dataset:
        print(data[0].size())
        print(data[1])
