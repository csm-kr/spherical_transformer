import glob
import os
import numpy as np
from collections import defaultdict
from datasets.sun360_icosahedron import PanoDatasetTrain, PanoDatasetVal

# get top10 classes
# dataset
# read top 10 classes
# dict[class_code] = list of (path, image (np.array)) (sorted in alphabetical order)
# dataset (train, validation)

def _get_class_code(cls): # for windows
    cls = cls.split('\\')
    return cls[1]+'/'+cls[2]

def _get_image_paths(root_dir):
    indoor_classes = glob.glob(os.path.join(root_dir, 'indoor/*'))
    outdoor_classes = glob.glob(os.path.join(root_dir, 'outdoor/*'))
    img_info = []
    img_dict = dict()
    gt_dict = dict()
    for cls in indoor_classes+outdoor_classes:
        if 'other' in cls:
            continue
        image_paths = glob.glob(cls+'/*.jpg')
        class_code = _get_class_code(cls)
        img_info.append((len(image_paths), class_code, image_paths))

    img_info = sorted(img_info, key = lambda x:x[0], reverse=True)
    for idx in range(10): # for top 10 common classes
        num_imgs, class_code, image_paths = img_info[idx]
        img_dict[class_code] = image_paths
        gt_dict[class_code] = idx
    return img_dict, gt_dict

def _sort_alphabetic(img_dict):
    for key in img_dict.keys():
        not_sorted = img_dict[key]
        sorted_paths = sorted(not_sorted, key=lambda x: x.split('\\')[-1])
        img_dict[key] = sorted_paths
    return img_dict

def _split_img_dict(img_dict, num_k, order):
    train_img_dict = defaultdict(list)
    val_img_dict = dict()
    for key in img_dict.keys():
        array_splitted = np.array_split(img_dict[key], num_k)
        val_img_dict[key] = list(array_splitted[order])
        # del array_splitted[order]
        for idx, arr in enumerate(array_splitted):
            if idx == order:
                continue
            train_img_dict[key].extend(list(array_splitted[idx]))
    # additional assertion check
    for key in train_img_dict.keys():
        intersection = set(train_img_dict[key]).intersection(set(val_img_dict[key]))
        assert len(intersection) == 0

    return train_img_dict, val_img_dict

# if __name__ == '__main__':
#     img_dict = _get_image_paths('E:/Downloads/SUN360_panoramas_1024x512/pano1024x512')
#     for key in img_dict.keys():
#         print(img_dict[key])
#         break
#     print('----------------------after sort----------------------')
#     img_dict = _sort_alphabetic(img_dict)
#     for key in img_dict.keys():
#         print(img_dict[key])
#         break
#     exit()
import cv2.cuda
def generate_train_val_set(division_level=5, num_k=3, order=1):
    root_dir = 'E:/Downloads/SUN360_panoramas_1024x512/pano1024x512'
    img_dict, gt_dict = _get_image_paths(root_dir) # get top 10 images dict[class code] = [img path...]
    for key in img_dict.keys():
        print(key, len(img_dict[key]))
    exit()
    img_dict = _sort_alphabetic(img_dict)
    train_img_dict, val_img_dict = _split_img_dict(img_dict, num_k, order)
    train_set = PanoDatasetTrain(division_level, train_img_dict, gt_dict, num_rotation = 5000)
    val_set = PanoDatasetVal(division_level, val_img_dict, gt_dict, num_rotation_per_image= 10, random_seed = 32)
    return train_set, val_set

if __name__ == '__main__':
    generate_train_val_set(division_level=5, num_k = 3, order=1) # order = [0,1,2]