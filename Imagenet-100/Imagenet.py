
from __future__ import print_function
import os
import os.path
import errno
import numpy as np
import sys
import cv2
from PIL import Image

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(class_file):
    with open(class_file) as r:
        classes = list(map(lambda s : s.strip(), r.readlines()))

    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx

def loadPILImage(path):
    trans_img = Image.open(path).convert('RGB')
    return trans_img

def loadCVImage(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    trans_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(trans_img.astype('uint8'), 'RGB')

def make_dataset(root, base_folder, dirname, class_to_idx):
    images = []
    dir_path = os.path.join(root, base_folder, dirname)

    if dirname == 'train':
        for fname in sorted(os.listdir(dir_path)):
            cls_fpath = os.path.join(dir_path, fname)
            if os.path.isdir(cls_fpath):
                cls_imgs_path = os.path.join(cls_fpath, 'images')
                for imgname in sorted(os.listdir(cls_fpath)):
                    if is_image_file(imgname):
                        path = os.path.join(cls_fpath, imgname)
                        item = (path, class_to_idx[fname])
                        images.append(item)
    else:
        for fname in sorted(os.listdir(dir_path)):
            cls_fpath = os.path.join(dir_path, fname)
            if os.path.isdir(cls_fpath):
                cls_imgs_path = os.path.join(cls_fpath, 'images')
                for imgname in sorted(os.listdir(cls_fpath)):
                    if is_image_file(imgname):
                        path = os.path.join(cls_fpath, imgname)
                        item = (path, class_to_idx[fname])
                        images.append(item)

        '''
        imgs_path = os.path.join(dir_path, 'images')
        imgs_annotations = os.path.join(dir_path, 'val_annotations.txt')

        with open(imgs_annotations) as r:
            data_info = map(lambda s : s.split('\t'), r.readlines())

        cls_map = {line_data[0]: line_data[1] for line_data in data_info}

        for imgname in sorted(os.listdir(imgs_path)):
            if is_image_file(imgname):
                path = os.path.join(imgs_path, imgname)
                item = (path, class_to_idx[cls_map[imgname]])
                images.append(item)
        '''
    return images

class Imagenet100(data.Dataset):

    base_folder = 'Imagenet-100'

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, loader = 'opencv'):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.loader = loader

        if download:
            self.download()

        _, class_to_idx = find_classes(os.path.join(self.root, self.base_folder, 'wnids.txt'))
        # self.classes = classes

        if self.train:
            dirname = 'train'
        else:
            dirname = 'val'

        self.data_info = make_dataset(self.root, self.base_folder, dirname, class_to_idx)

        if len(self.data_info) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (img_path, target) where target is index of the target class.
        """

        img_path, target = self.data_info[index][0], self.data_info[index][1]

        if self.loader == 'pil':
            img = loadPILImage(img_path)
        else:
            img = loadCVImage(img_path)

        if self.transform is not None:
            result_img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return result_img, target

    def __len__(self):
        return len(self.data_info)

