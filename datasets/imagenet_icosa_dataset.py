import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.imagenet import load_meta_file
from torchvision.datasets.folder import ImageFolder

from samplings.cube_sampling import inflate_cube
from samplings.icosahedron_sampling import inflate_icosahedron

from utils.visualization_util import show_spheres, grid_2_points
from utils.projection_util import get_projection_grid, rotate_grid, cartesian_to_spherical, spherical_to_plane, make_fov_projection_map


class ImageNet_Icosa_Dataset(ImageFolder):

    def __init__(self,
                 root: str,
                 split: str,
                 is_minival: bool = False,
                 download: bool = False,
                 rotate: bool = True,
                 vis: bool = False,
                 bandwidth: int = 100,
                 division_level: int = 58
                 ):

        self.root = root
        assert split in ('train', 'val')
        self.split = split  # training set or test set
        self.is_minival = is_minival

        wnid_to_classes = load_meta_file(self.root)[0]

        super(ImageNet_Icosa_Dataset, self).__init__(self.split_folder)
        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx
                             for idx, clss in enumerate(self.classes)
                             for cls in clss}

        self.bandwidth = bandwidth
        self.division_level = division_level
        self.rotate = rotate
        self.vis = vis
        self.omni_h = self.omni_w = self.bandwidth * 2

        self.phi_fov = 65.5
        self.theta_fov = 65.5
        self.fov_proj_map_x, self.fov_proj_map_y = \
            make_fov_projection_map(self.omni_h, self.omni_w, self.phi_fov, self.theta_fov)

        if rotate and split == 'val':
            # fix for testing
            np.random.seed(7788)
            # 50000 개 중에 50000 개
            num_rotations = 50000
            num_val_img = 50000
            self.test_rot_idx = np.random.choice(num_rotations, num_val_img)

        # make minival samples
        train_samples = []
        minival_samples = []

        label_before = -1
        for i, sample in enumerate(self.samples):
            label_now = sample[1]
            if label_before != label_now:
                # print('label_now : {}, label_before : {}'.format(label_now, label_before))
                label_before += 1
                cnt = 0
            if cnt < 50:
                minival_samples.append(sample)
            else:
                train_samples.append(sample)
            cnt += 1

        assert len(self.samples) == len(train_samples) + len(minival_samples)

        if self.split == 'train' and self.is_minival:
            self.samples = minival_samples
        elif self.split == 'train' and not self.is_minival:
            self.samples = train_samples

        self.cube_face_list = inflate_icosahedron(division_level=division_level)
        self.cube_mapping_list = []

        # make mapping matrix for icosahedron to erp
        # loop cube face (20)
        for icosa_face in self.icosa_face_list:
            num_point = icosa_face.shape[0]  # num point, 3
            icosa_sampling_map_x = np.zeros(num_point, dtype=np.float32)
            icosa_sampling_map_y = np.zeros(num_point, dtype=np.float32)

            # each points convert cartesian(x, y, z) to spherical(phi, theta)
            for n_i in range(num_point):
                [p, t] = cartesian_to_spherical(icosa_face[n_i, 0], icosa_face[n_i, 1], icosa_face[n_i, 2])
                y_on_equi, x_on_equi = spherical_to_plane(p, t, self.omni_h * 3, self.omni_w * 3)
                icosa_sampling_map_x[n_i] = x_on_equi
                icosa_sampling_map_y[n_i] = y_on_equi
            self.icosa_mapping_list.append((icosa_sampling_map_x, icosa_sampling_map_y))


    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, self.split)

    def __getitem__(self, idx):

        path, target = self.samples[idx]
        img = self.loader(path)
        img_np = np.array(img)
        img_np = cv2.resize(img_np, (self.omni_h, self.omni_w))

        fov_proj_map_x = self.fov_proj_map_x
        fov_proj_map_y = self.fov_proj_map_y

        undefined_zone = fov_proj_map_y == -1
        OMNI_H, OMNI_W = fov_proj_map_x.shape
        img_np = cv2.remap(img_np, fov_proj_map_x, fov_proj_map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
        # [549, 549]
        img_np[undefined_zone] = 0

        # for erp projection sampling -> resize
        img_np = cv2.resize(img_np, (self.omni_h * 3, self.omni_w * 3))  # [600, 600]

        if self.rotate:
            # get random index at training
            if self.split == 'train':
                rot_idx = np.random.randint(0, 50000)
            # get fixed index at testing
            elif self.split == 'val':
                rot_idx = self.test_rot_idx[idx]

            # -------------------------------------- choose one method for remap --------------------------------------
            #  ################### 1) create rotation remap ###################

            # R = rand_rotation_matrix()
            # map_x, map_y = rotate_map_given_R(R, self.omni_h, self.omni_w)
            # img_np = cv2.remap(img_np, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)

            #  ################### 2) load rotation remap ###################
            now_dir = os.getcwd()

            # for dataset test
            # map_path_name = 'xy_maps_50000_mnist'  # 'xy_maps_50_50'
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
        return len(self.samples)


if __name__ == '__main__':
    dataset = ImageNet_Icosa_Dataset(root='D:\data\ILSVRC_classification', split='val', rotate=True, vis=True)
    print(len(dataset))
    img, label = dataset.__getitem__(0)
    print(img.shape)