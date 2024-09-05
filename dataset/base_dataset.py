import os
from typing import Dict, Any

import numpy as np
import torchvision as tv
from torch.utils.data import Dataset
from PIL import Image
from .register import dataset_entrypoints
from .utils.filter_images import filter_images, save_list_from_filter, load_list_from_path


# 这里存放的是dataset的模板，继承这个模板实现相应的功能就可以了

class BaseSegmentation(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, need_index_name=True, classes=None,
                 ignore_index=None):
        if ignore_index is None:
            # 一般target中255都是忽略的地方（黑色背景）
            self.ignore_index = [255]
        else:
            self.ignore_index = ignore_index
        self.root = root
        self._check_path_exists(root)
        if transform is None:
            self.transform = tv.transforms.ToTensor()
        else:
            self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.need_index_name = need_index_name
        if need_index_name and classes is None:
            raise ValueError("you have to specify the classes when need index_name")
        self.classes = classes
        self._modify_classes_dict()
        splits_file = self._get_path()
        self._check_path_exists(splits_file)
        self.images = self._load_data_path_to_list(splits_file)

    @staticmethod
    def _check_path_exists(path):
        # 检测是否存在数据集
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'path not found or corrupted and the path is {path}'
            )

    def _get_path(self):
        """这个类需要被重写，引导到储存文件图片路径的文档，默认是root_dir下list中train.txt"""
        return os.path.join(self.root, "list", 'train.txt')

    def _load_data_path_to_list(self, path):
        """如果这里的文件是一行中前面是相对于root_dir的image path，后面是target path，例如：JPEGImages/2007_000032.jpg，那么就不用重写"""
        images = []
        with open(path, 'r') as f:
            for line in f:
                x = line.strip().split(",")
                images.append((os.path.join(self.root, x[0]), os.path.join(self.root, x[1])))
        return images

    def apply_new_data_list(self, new_data_list_path):
        self._check_path_exists(new_data_list_path)
        self.images = self._load_data_path_to_list(new_data_list_path)

    def _get_text_prompt_from_target(self, target):
        """
        这里默认target中简单通过灰度值储存label，例如label序号是3则图片对应位置灰度值是3，如果是彩色target需要重写此方法
        同时这里读取label默认读取全部label，如果有label对应空或者背景最好重写剔除
        """
        unique_values = np.unique(np.array(target).flatten())
        target_text = [self.classes[x] for x in unique_values if x not in self.ignore_index]
        text_prompt = ".".join(target_text)
        return text_prompt

    def get_class_index(self):
        return list(self.classes.keys())

    def _modify_classes_dict(self):
        for i in self.ignore_index:
            self.classes[i] = "ignore"

    def __getitem__(self, index):
        image = Image.open(self.images[index][0]).convert('RGB')
        target = Image.open(self.images[index][1])
        if self.transform is not None:
            image, target = self.transform(image, target)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.need_index_name:
            text_prompt = self._get_text_prompt_from_target(target)
            return {"data": (image, target), "path": (self.images[index][0], self.images[index][1]),
                    "text_prompt": text_prompt}
        return {"data": (image, target), "path": (self.images[index][0], self.images[index][1])}

    def __len__(self):
        return len(self.images)


class BaseIncrement(Dataset):

    def __init__(self, segmentation_dataset_name=None, segmentation_config=None, train=True,
                 labels=None, labels_old=None, overlap=True, masking=True, data_masking="current", no_memory=True,
                 new_image_path=None, save_stage_image_list_path=None,mask_value=255):
        self.no_memory = no_memory
        if not self.no_memory:
            raise NotImplementedError("not implemented")

        self.dataset = dataset_entrypoints(segmentation_dataset_name)(**segmentation_config)
        self.ignore_index = self.dataset.ignore_index
        self.__strip_ignore(labels)
        self.__strip_ignore(labels_old)
        assert not any(i in labels_old for i in labels)  # 排除忽略的index以后，之前stage训练的label和当前stage训练的label要互斥

        if new_image_path is not None and os.path.exists(new_image_path):
            idx = load_list_from_path(new_image_path)
        else:
            idx = filter_images(self.dataset, labels, labels_old, overlap=overlap)
            if save_stage_image_list_path is not None:
                save_list_from_filter(idx, save_stage_image_list_path)
        
        self.dataset.images = idx
        
        self.train = train
        
        self.order = self.dataset.get_class_index()
        self.labels = labels
        self.labels_old = labels_old
        self.data_masking = data_masking
        self.overlap = overlap
        self.masking = masking
        self.mask_value = mask_value
        self._create_inverted_order()
        self.dataset.target_transform = self._create_target_transform()

    def __strip_ignore(self, labels):
        for i in self.ignore_index:
            while i in labels:
                labels.remove(i)

    def _create_new_path_list(self):
        """这里要重写，给数据集按照要求创建新的索引表"""
        pass

    def _create_inverted_order(self, mask_value=255):
        # 映射label和索引
        self.inverted_order = {label: self.order.index(label) for label in self.order if label not in self.ignore_index}

    def _create_target_transform(self):
        reorder_transform = tv.transforms.Lambda(
            lambda t: t.apply_(
                lambda x: self.inverted_order[x] if x in self.inverted_order else masking_value
            )
        )

        if self.masking:
            if self.data_masking == "current":
                tmp_labels = self.labels + [255]
            elif self.data_masking == "current+old":
                tmp_labels = self.labels_old + self.labels + [255]
            elif self.data_masking == "all":
                raise NotImplementedError(
                    f"data_masking={self.data_masking} not yet implemented sorry not sorry."
                )
            elif self.data_masking == "new":
                tmp_labels = self.labels
                masking_value = 255

            target_transform = tv.transforms.Lambda(
                lambda t: t.
                apply_(lambda x: self.inverted_order[x] if x in tmp_labels else masking_value)
            )
        else:
            target_transform = reorder_transform
            assert False
        return target_transform

    def __getitem__(self, index):
        return self.dataset[index]
