import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.imagenet import load_meta_file
from torchvision.datasets.folder import ImageFolder

from utils.visualization_util import show_spheres, grid_2_points
from utils.projection_util import get_projection_grid, rotate_grid, make_fov_projection_map
from utils.rotation_util import calculate_Rmatrix_from_phi_theta, rotate_map_given_R


class ImageNet_ERP_Dataset(ImageFolder):

    def __init__(self,
                 root: str,
                 split: str,
                 is_minival: bool = False,
                 download: bool = False,
                 rotate: bool = True,
                 vis: bool = False,
                 bandwidth: int = 100,
                 ):

        self.root = root
        assert split in ('train', 'val')
        self.split = split  # training set or test set
        self.is_minival = is_minival
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


    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, self.split)

    def __getitem__(self, idx):

        path, target = self.samples[idx]
        img = self.loader(path)
        img_np = np.array(img)
        img_np = cv2.resize(img_np, (self.omni_h, self.omni_w))

        # D-H ERP
        # -------------------------------- make fov proj --------------------------------
        # fov_proj_map_x, fov_proj_map_y, = make_fov_projection_map(self.omni_h * 3, self.omni_w * 3, self.phi_fov, self.theta_fov)

        # -------------------------------- load fov proj --------------------------------
        # now_dir = os.getcwd()
        # # map_path_name = 'xy_fov_maps_200'
        # map_path_name = r'D:\data\\xy_maps_50000_image_600'
        # if 'datasets' in now_dir.split('\\'):
        #     map_matrix_dir = os.path.join(os.path.split(now_dir)[0], map_path_name)
        # else:
        #     map_matrix_dir = os.path.join(now_dir, map_path_name)
        #
        # map_x_path = map_matrix_dir + '/' + 'x.npy'
        # map_y_path = map_matrix_dir + '/' + 'y.npy'
        # fov_proj_map_x = np.load(map_x_path)
        # fov_proj_map_y = np.load(map_y_path)

        fov_proj_map_x = self.fov_proj_map_x
        fov_proj_map_y = self.fov_proj_map_y

        undefined_zone = fov_proj_map_y == -1
        OMNI_H, OMNI_W = fov_proj_map_x.shape
        img_np = cv2.remap(img_np, fov_proj_map_x, fov_proj_map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)   # [600, 600]
        img_np[undefined_zone] = 0

        # for erp projection sampling -> resize
        img_np = cv2.resize(img_np, (self.omni_h * 3, self.omni_w * 3))

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
        img_np = cv2.resize(img_np, (self.bandwidth * 2, self.bandwidth))

        if self.vis:

            grid = get_projection_grid(b=self.bandwidth, grid_type='ERP')
            points = grid_2_points(grid=grid)  # tuples -> (num_points, 3)

            img_np_vis = img_np
            cv2.imshow('rotated_img', img_np_vis)
            cv2.waitKey(0)
            rgb = img_np_vis.reshape(-1, 3)                             # [H * W, 3]

            show_spheres(scale=2, points=points, rgb=rgb)       # points, rgb : (num_points, 3)

        img_np = np.transpose(img_np, (2, 0, 1)).astype(np.float32) / 255          # [0 ~ 1]
        img_torch = torch.FloatTensor(img_np)                   # [1, 60, 60]

        label = int(self.targets[idx])
        return img_torch, label

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    dataset = ImageNet_ERP_Dataset(root='D:\data\ILSVRC_classification', split='val', rotate=True, vis=True)
    print(len(dataset))
    img, label = dataset.__getitem__(11)