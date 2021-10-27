import os
import cv2
import torch
import numpy as np
import glob

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.imagenet import load_meta_file
from torchvision.datasets.folder import ImageFolder

from utils.visualization_util import show_spheres, grid_2_points
from utils.projection_util import get_projection_grid, rotate_grid, make_fov_projection_map
from utils.rotation_util import calculate_Rmatrix_from_phi_theta, rotate_map_given_R


class Panoramic_ERP_Dataset(Dataset):

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
        self.omni_h = self.omni_w = self.bandwidth * 2

        # if rotate and split == 'val':
        #     # fix for testing
        #     np.random.seed(7788)
        #     # 50000 개 중에 50000 개
        #     num_rotations = 50000
        #     num_val_img = 50000
        #     self.test_rot_idx = np.random.choice(num_rotations, num_val_img)

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

    def __getitem__(self, idx):

        img = cv2.imread(self.img_path[idx])

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

        return img_torch, label

    def __len__(self):
        return len(self.img_path)


if __name__ == '__main__':
    dataset = Panoramic_ERP_Dataset(root='D:\data\panorama_360', split='train', rotate=True, vis=True)
    print(len(dataset))
    img, label = dataset.__getitem__(0)