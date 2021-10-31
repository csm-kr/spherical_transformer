# torch
import torch
from torch.utils.data import Dataset, DataLoader
import os
import warnings
# cv
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
# utils
# from utils import download_and_extract_archive, read_label_file, read_image_file, show_spheres
# from util.projection_utils import make_gnomonic_projection_map
# from util.rotation_utils import cartesian_to_spherical, rotate_image_given_phi_theta_efficient
# from util.utils import spherical_to_plane
# sampling
# from icosahedron_inflating import inflate_icosahedron
import random
import glob
from collections import defaultdict
from tqdm import tqdm
import time
from PIL import Image
import json
# from precompute_LUT import compute_LUT


import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.visualization_util import show_spheres, grid_2_points
from utils.projection_util import get_projection_grid, rotate_grid, make_fov_projection_map, spherical_to_plane
from utils.rotation_util import calculate_Rmatrix_from_phi_theta, rotate_map_given_R, rotate_image_given_phi_theta_efficient
from utils.rotation_util import cartesian_to_spherical
from samplings.icosahedron_sampling import inflate_icosahedron
from datasets.precompute_LUT import compute_LUT, build_rotation_for_PanoVal


class PanoDatasetTrain(Dataset):
    # This class is for train set
    def _load_images(self, img_dict):
        self.img_dict = defaultdict(list)
        self.num_data = 0
        for key in img_dict.keys():
            for img_p in img_dict[key]:
                self.img_dict[key].append((img_p, cv2.imread(img_p)))
                self.num_data += 1

    def __init__(self, division_level, img_dict, gt_dict, num_rotation=10, visualization=False):
        super(PanoDatasetTrain, self).__init__()
        # self.root_dir = '/home/yhat/PycharmProjects/SUN360_panoramas_1024x512/pano1024x512'
        self.division_level = division_level
        self.visualization = visualization
        self._load_images(img_dict)

        self.gt_dict = gt_dict
        self.class_codes = [key for key in self.gt_dict.keys()]

        # build R matrix
        self.omni_h = 512
        self.omni_w = 512
        self.R_list = compute_LUT(num_rotation, self.omni_h, self.omni_w) #(num_R, H, W)

        self.sphere_phd_face_list = inflate_icosahedron(self.division_level)
        self.sampling_matrix_tuple_list = []
        for sphere_phd_face in self.sphere_phd_face_list:
            num_point = sphere_phd_face.shape[0]  # num point, 3
            icosa_sampling_map_x = np.zeros(num_point, dtype=np.float32)
            icosa_sampling_map_y = np.zeros(num_point, dtype=np.float32)
            for n_i in range(num_point):
                [p, t] = cartesian_to_spherical(sphere_phd_face[n_i, 0], sphere_phd_face[n_i, 1],
                                                sphere_phd_face[n_i, 2])
                y_on_equi, x_on_equi = spherical_to_plane(p, t, self.omni_h, self.omni_w)
                icosa_sampling_map_x[n_i] = x_on_equi
                icosa_sampling_map_y[n_i] = y_on_equi
            self.sampling_matrix_tuple_list.append((icosa_sampling_map_x, icosa_sampling_map_y))

    def __getitem__(self, index):

        class_code = random.choice(self.class_codes)
        (path, equi_img) = random.choice(self.img_dict[class_code])

        map_dict = random.choice(self.R_list)
        map_x = map_dict['map_x']
        map_y = map_dict['map_y']
        rotated_equi = cv2.remap(equi_img, map_x, map_y, cv2.INTER_CUBIC)

        coordinates = []
        patch_list = []
        for coord, maps in zip(self.sphere_phd_face_list, self.sampling_matrix_tuple_list):
            map_x = maps[0]
            map_y = maps[1]
            # sanity check
            # for x_c, y_c in zip(map_x, map_y):
            #     cv2.circle(rotated_equi, (x_c.astype(np.int32), y_c.astype(np.int32)), 1, (0, 255, 0), -1)
            bert_input_patch = cv2.remap(rotated_equi, map_x, map_y, cv2.INTER_CUBIC)  # [16, 1]
            coordinates.append(coord)
            patch_list.append(bert_input_patch)

        coordinates = np.array(coordinates)  # 20(num_face), 16(4 ** division_level), 3   (x, y, z)
        patch_list = np.array(patch_list)    # 20(num_face), 16(4 ** division_level), 1/3 (gray/rgb)

        # if self.visualization:
        #     coordinates_vis = coordinates.reshape(20 * 4 ** self.division_level, -1)  # [320, 3]
        #     cal_vis = patch_list.reshape(20 * 4 ** self.division_level, -1)           # [320, 1] (gray_scale)
        #     plt.imshow(img)
        #     show_spheres(scale=2, points=coordinates_vis, rgb=cal_vis)

        sequence_image = torch.from_numpy(patch_list).type(torch.float32)
        sequence_image = sequence_image.reshape(20, -1)
        target = torch.tensor(self.gt_dict[class_code]).squeeze(-1)
        target = target.long()
        return sequence_image, target

    def __len__(self):
        return self.num_data


class PanoDatasetVal(Dataset):
    # This class is for train set
    def _load_images(self, val_img_dict):
        num_imgs = 0
        for key in val_img_dict.keys():
            num_imgs += len(val_img_dict[key])
        num_total_R = num_imgs*self.num_rotation_per_image
        p_list = np.random.uniform(0, 180, size=num_total_R)
        t_list = np.random.uniform(0, 360, size=num_total_R)
        R_list = build_rotation_for_PanoVal(p_list, t_list, self.omni_h, self.omni_w)
        img_list = list()
        cnt = 0
        for key in val_img_dict.keys():
            for img_p in val_img_dict[key]:
                img = cv2.imread(img_p)
                map_dict = R_list[cnt]
                rotated_img = cv2.remap(img, map_dict['map_x'], map_dict['map_y'], cv2.INTER_CUBIC)
                cv2.imshow('sanity check for validation dataset', rotated_img)
                cv2.waitKey(0)
                img_list.append((rotated_img, key))
                cnt+=1
        self.num_data = cnt
        return img_list

    def __init__(self, division_level, val_img_dict, gt_dict, num_rotation_per_image, random_seed):
        super(PanoDatasetVal, self).__init__()
        self.division_level = division_level
        self.random_seed = random_seed
        self.num_rotation_per_image = num_rotation_per_image
        # self.visualization = visualization
        # self.data_dict = defaultdict(list)
        self._load_images(val_img_dict)
        self.gt_dict = gt_dict
        self.class_codes = [key for key in self.gt_dict.keys()]

        self.omni_h = 512
        self.omni_w = 512
        self.img_list = self._load_images()

        self.sphere_phd_face_list = inflate_icosahedron(self.division_level)
        self.sampling_matrix_tuple_list = []
        for sphere_phd_face in self.sphere_phd_face_list:
            num_point = sphere_phd_face.shape[0]  # num point, 3
            icosa_sampling_map_x = np.zeros(num_point, dtype=np.float32)
            icosa_sampling_map_y = np.zeros(num_point, dtype=np.float32)
            for n_i in range(num_point):
                [p, t] = cartesian_to_spherical(sphere_phd_face[n_i, 0], sphere_phd_face[n_i, 1],
                                                sphere_phd_face[n_i, 2])
                y_on_equi, x_on_equi = spherical_to_plane(p, t, self.omni_h, self.omni_w)
                icosa_sampling_map_x[n_i] = x_on_equi
                icosa_sampling_map_y[n_i] = y_on_equi
            self.sampling_matrix_tuple_list.append((icosa_sampling_map_x, icosa_sampling_map_y))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        rotated_equi, class_code = self.img_list[index]

        # map_dict = random.choice(self.R_list)
        # map_x = map_dict['map_x']
        # map_y = map_dict['map_y']
        # rotated_equi = cv2.remap(equi_img, map_x, map_y, cv2.INTER_CUBIC)

        coordinates = []
        patch_list = []
        for coord, maps in zip(self.sphere_phd_face_list, self.sampling_matrix_tuple_list):
            map_x = maps[0]
            map_y = maps[1]
            bert_input_patch = cv2.remap(rotated_equi, map_x, map_y, cv2.INTER_CUBIC)  # [16, 1]
            coordinates.append(coord)
            patch_list.append(bert_input_patch)

        coordinates = np.array(coordinates)  # 20(num_face), 16(4 ** division_level), 3   (x, y, z)
        patch_list = np.array(patch_list)    # 20(num_face), 16(4 ** division_level), 1/3 (gray/rgb)

        # if self.visualization:
        #     coordinates_vis = coordinates.reshape(20 * 4 ** self.division_level, -1)  # [320, 3]
        #     cal_vis = patch_list.reshape(20 * 4 ** self.division_level, -1)           # [320, 1] (gray_scale)
        #     plt.imshow(img)
        #     show_spheres(scale=2, points=coordinates_vis, rgb=cal_vis)

        sequence_image = torch.from_numpy(patch_list).type(torch.float32)
        sequence_image = sequence_image.reshape(20, -1)
        target = torch.tensor(self.gt_dict[class_code]).squeeze(-1)
        target = target.long()

        return sequence_image, target

    def __len__(self):
        return self.num_data



class PanoDataset_val_deterministic(Dataset):
    def _load_images(self):
        self.num_data = 0
        indoor_classes = glob.glob(self.root_dir + '/' + 'indoor/*_____'+self.type)
        outdoor_classes = glob.glob(self.root_dir + '/' + 'outdoor/*_____'+self.type)
        img_paths_list = []

        for idx, cls in tqdm(enumerate(indoor_classes + outdoor_classes)):
            img_paths = glob.glob(cls+'/*.jpg')
            cls = cls.split('_____'+self.type)[0]
            class_code = cls.split('/')[-2] + '/' + cls.split('/')[-1]
            for img_p in img_paths:
                img = cv2.imread(img_p)
                for ii in range(self.num_rotation):
                    img_paths_list.append((img_p, img, class_code, ii))
                    self.num_data += 1

        return img_paths_list

    def __init__(self, division_level=5, num_rotation=2000, visualization=False, type='minival'):
        super(PanoDataset_val_deterministic, self).__init__()
        assert type in ['minival' ,'test']
        self.type = type
        self.root_dir = '/home/yhat/PycharmProjects/SUN360_panoramas_1024x512/pano1024x512'
        self.division_level = division_level
        self.visualization = visualization
        self.data_dict = defaultdict(list)
        self.num_rotation = num_rotation

        self.gt_dict = gt_dict
        self.class_codes = [key for key in self.gt_dict.keys()]
        self.img_paths_list = self._load_images()

        # build R matrix
        self.omni_h = 512
        self.omni_w = 512
        self.R_list = compute_LUT(num_rotation, self.omni_h, self.omni_w) #(num_R, H, W)

        self.gt_keys = list(self.data_dict.keys())
        self.sphere_phd_face_list = inflate_icosahedron(self.division_level)
        self.sampling_matrix_tuple_list = []
        for sphere_phd_face in self.sphere_phd_face_list:
            num_point = sphere_phd_face.shape[0]  # num point, 3
            icosa_sampling_map_x = np.zeros(num_point, dtype=np.float32)
            icosa_sampling_map_y = np.zeros(num_point, dtype=np.float32)
            for n_i in range(num_point):
                [p, t] = cartesian_to_spherical(sphere_phd_face[n_i, 0], sphere_phd_face[n_i, 1],
                                                sphere_phd_face[n_i, 2])
                y_on_equi, x_on_equi = spherical_to_plane(p, t, self.omni_h, self.omni_w)
                icosa_sampling_map_x[n_i] = x_on_equi
                icosa_sampling_map_y[n_i] = y_on_equi
            self.sampling_matrix_tuple_list.append((icosa_sampling_map_x, icosa_sampling_map_y))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        (img_p, equi_img, class_code, rotation_idx) = self.img_paths_list[index]
        # print(f'class code {class_code} num imgs in img_dict {len(self.img_dict[class_code])}')
        # (path, equi_img) = random.choice(self.img_dict[class_code])
        map_dict = self.R_list[rotation_idx]
        map_x = map_dict['map_x']
        map_y = map_dict['map_y']
        rotated_equi = cv2.remap(equi_img, map_x, map_y, cv2.INTER_CUBIC)

        coordinates = []
        patch_list = []
        for coord, maps in zip(self.sphere_phd_face_list, self.sampling_matrix_tuple_list):
            map_x = maps[0]
            map_y = maps[1]
            bert_input_patch = cv2.remap(rotated_equi, map_x, map_y, cv2.INTER_CUBIC)  # [16, 1]
            coordinates.append(coord)
            patch_list.append(bert_input_patch)

        coordinates = np.array(coordinates)  # 20(num_face), 16(4 ** division_level), 3   (x, y, z)
        patch_list = np.array(patch_list)    # 20(num_face), 16(4 ** division_level), 1/3 (gray/rgb)

        # if self.visualization:
        #     coordinates_vis = coordinates.reshape(20 * 4 ** self.division_level, -1)  # [320, 3]
        #     cal_vis = patch_list.reshape(20 * 4 ** self.division_level, -1)           # [320, 1] (gray_scale)
        #     plt.imshow(img)
        #     show_spheres(scale=2, points=coordinates_vis, rgb=cal_vis)

        sequence_image = torch.from_numpy(patch_list).type(torch.float32)
        sequence_image = sequence_image.reshape(20, -1)
        target = torch.tensor(self.gt_dict[class_code]).squeeze(-1)
        target = target.long()

        return sequence_image, target

    def __len__(self):
        return self.num_data

if __name__ == '__main__':

    dataset = PanoDataset(division_level=5, visualization=False)
    output = next(iter(dataset))

    print(output)
    # for img, target in(dataloader):
    #     print(img.size())
    #     print(target.size())