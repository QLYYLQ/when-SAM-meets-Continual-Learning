import os
from typing import Optional, List, Callable
import numpy as np
import torchvision as tv
from torch.utils.data import Dataset
from PIL import Image
from .register import dataset_entrypoints
from .utils.filter_images import filter_images, save_list_from_filter, load_list_from_path
from torch.utils.data import Dataset,DataLoader

# 这里存放的是dataset的模板，继承这个模板实现相应的功能就可以了

class BaseSplit(Dataset):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 need_index_name: bool = True,
                 classes: Optional[dict] = None,
                 ignore_index: Optional[List] = None,
                 mask_value: int = 255):

        if ignore_index is None:
            # 一般target中255都是忽略的地方（黑色背景）
            self.ignore_index = [255]
        else:
            self.ignore_index = ignore_index if 255 in ignore_index else ignore_index+[255]
        self.root = root
        self.mask_value = mask_value
        self._check_path_exists(root)
        self.is_filter = False
        if not transform:
            self._init_image_transform()
        else:
            self.transform = transform
        if not target_transform:
            self.target_transform = self._init_target_transform
        else:
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
    def _init_target_transform(self,image):
        """把一些需要忽略掉的label对应的target换成255"""
        image = np.array(image)
        mask = np.isin(image, self.ignore_index)
        image[mask] = self.mask_value
        return image

    def _init_image_transform(self):
        self.transform = tv.transforms.Compose([tv.transforms.ToTensor()])

    def _get_path(self):
        """这个类需要被重写，引导到储存文件图片路径的文档，默认是root_dir下list中train.txt"""
        return os.path.join(self.root, "list", 'train.txt')

    def _load_data_path_to_list(self, path):
        """如果这里的文件是一行中前面是相对于root_dir的image path，后面是target path，例如：JPEGImages/2007_000032.jpg，那么就不用重写"""
        images = []
        with open(path, 'r') as f:
            for line in f:
                x = line.strip().split(",")
                images.append((os.path.join(self.root, x[0]).replace(os.sep, "/"),
                               os.path.join(self.root, x[1]).replace(os.sep, "/")))
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
        return [x for x in self.classes.keys() if x not in self.ignore_index]

    def _modify_classes_dict(self):
        for i in self.ignore_index:
            self.classes[i] = "ignore"

    def __getitem__(self, index):
        single_batch={}
        single_batch['path']=(self.images[index][0], self.images[index][1])
        image = Image.open(self.images[index][0]).convert('RGB')
        target = Image.open(self.images[index][1])
        if not self.is_filter:
            if self.need_index_name:
                text_prompt = self._get_text_prompt_from_target(target)+"."
                single_batch["text_prompt"] = text_prompt
            if self.target_transform is not None:
                target = self.target_transform(target)
            if self.transform is not None:
                image = self.transform(image)
                target = self.transform(target)
            single_batch["data"]=(image, target)
            return single_batch
        else:
            single_batch["data"] = (image, target)
            return single_batch

    def __len__(self):
        return len(self.images)


# class BaseIncrement(Dataset):
#
#     def __init__(self,
#                  segmentation_dataset_name: str = None,
#                  segmentation_config: dict = None,
#                  train: bool = True,
#                  labels: list = None,
#                  labels_old: list = None,
#                  overlap: bool = True,
#                  masking: bool = True,
#                  data_masking: str = "current",
#                  no_memory: bool = True,
#                  new_image_path: str = None,
#                  save_stage_image_list_path: str = None,
#                  mask_value: int = 255):
#         self.no_memory = no_memory
#         if not self.no_memory:
#             raise NotImplementedError("not implemented")
#
#         self.dataset = dataset_entrypoints(segmentation_dataset_name)(**segmentation_config)
#         self.ignore_index = self.dataset.ignore_index
#         self.__strip_ignore(labels)
#         self.__strip_ignore(labels_old)
#         assert not any(i in labels_old for i in labels)  # 排除忽略的index以后，之前stage训练的label和当前stage训练的label要互斥
#
#         if new_image_path is not None and os.path.exists(new_image_path):
#             idx = load_list_from_path(new_image_path)
#         else:
#             idx = filter_images(self.dataset, labels, labels_old, overlap=overlap)
#             if save_stage_image_list_path is not None:
#                 save_list_from_filter(idx, save_stage_image_list_path)
#
#         self.dataset.images = idx
#
#         self.train = train
#
#         self.order = self.dataset.get_class_index()
#         self.labels = labels
#         self.labels_old = labels_old
#         self.data_masking = data_masking
#         self.overlap = overlap
#         self.masking = masking
#         self.mask_value = mask_value
#         self._create_inverted_order()
#         self.dataset.target_transform = self._create_target_transform()
#
#     def __strip_ignore(self, labels):
#         for i in self.ignore_index:
#             while i in labels:
#                 labels.remove(i)
#
#     def _create_inverted_order(self, mask_value=255):
#         # 映射label和索引
#         self.inverted_order = {label: self.order.index(label) for label in self.order if label not in self.ignore_index}
#
#     def _create_target_transform(self):
#         mask_value = self.mask_value
#         target_transform = tv.transforms.Lambda(
#             lambda t: t.apply_(
#                 lambda x: x if x in self.order else mask_value
#             )
#         )
#
#         if self.masking:
#             if self.data_masking == "current":
#                 tmp_labels = self.labels + [mask_value]
#             elif self.data_masking == "current+old":
#                 tmp_labels = self.labels_old + self.labels + [255]
#             # elif self.data_masking == "all":
#             #     # 全部保留
#             #     target_transform = None
#             # # elif self.data_masking == "new":
#             # #     tmp_labels = self.labels
#             # #     masking_value = 255
#             #
#             target_transform = tv.transforms.Lambda(
#                 lambda t: t.
#                 apply_(lambda x: x if x in tmp_labels else mask_value)
#             )
#         return target_transform
#
#     def __getitem__(self, index):
#         return self.dataset[index]
#
#     def __len__(self):
#         return len(self.dataset)


class BaseIncrement(Dataset):
    def __init__(self,
                 split_dataset_name: str = None,
                 split_config: dict = None,
                 stage_index_dict: dict = None,
                 stage_path_dict: dict = None,
                 train: bool = True,
                 overlap: bool = True,
                 masking: bool = True,
                 data_masking: str = "current",
                 no_memory: bool = True,
                 mask_value: int = 255):
        self.no_memory = no_memory
        if not self.no_memory:
            raise NotImplementedError("not implemented")

        self.dataset = dataset_entrypoints(split_dataset_name)(**split_config)
        self.ignore_index = self.dataset.ignore_index
        self.stage_index_dict = stage_index_dict
        self.stage_path_dict = stage_path_dict
        self.labels = []
        self.labels_old = []
        self.stage = 0
        self.update_stage(0)

        self.__strip_ignore(self.labels)
        self.__strip_ignore(self.labels_old)
        assert not any(i in self.labels_old for i in self.labels)  # 排除忽略的index以后，之前stage训练的label和当前stage训练的label要互斥

        self.train = train

        self.order = self.dataset.get_class_index()
        self.data_masking = data_masking
        self.overlap = overlap
        self.masking = masking
        self.mask_value = mask_value
        self._create_inverted_order()
        self.dataset.target_transform.append(self._create_target_transform())

    def __strip_ignore(self, labels):
        for i in self.ignore_index:
            while i in labels:
                labels.remove(i)

    def _create_inverted_order(self, mask_value=255):
        # 映射label和索引
        self.inverted_order = {label: self.order.index(label) for label in self.order if label not in self.ignore_index}

    def _create_target_transform(self):
        mask_value = self.mask_value
        target_transform = tv.transforms.Lambda(
            lambda t: t.apply_(
                lambda x: x if x in self.order else mask_value
            )
        )
        if self.masking:
            if self.data_masking == "current":
                tmp_labels = self.labels + [mask_value]
            elif self.data_masking == "current+old":
                tmp_labels = self.labels_old + self.labels + [255]
            target_transform = tv.transforms.Lambda(
                lambda t: t.
                apply_(lambda x: x if x in tmp_labels else mask_value)
            )
        total_transform = tv.transforms.Compose([tv.transforms.ToTensor(), target_transform])
        return total_transform

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def update_stage(self, stage_number):
        max_stage = len(self.stage_index_dict.keys())
        labels_old = []
        if stage_number >= max_stage:
            raise ValueError("stage number out of range")
        if stage_number == 0:
            labels_old = []
            labels = self.stage_index_dict[stage_number]
        else:
            labels = self.stage_index_dict[stage_number]
            for i in range(stage_number):
                labels_old += self.stage_index_dict[i]
        self.labels = labels
        self.labels_old = labels_old
        self.dataset.apply_new_data_list(self.stage_path_dict[stage_number])



