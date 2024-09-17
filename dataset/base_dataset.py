import os
import numpy as np
import torch
import torchvision as tv
from typing import Optional, List, Callable, Tuple
from PIL import Image
from dataset.register import dataset_entrypoints
from torch.utils.data import Dataset
from typing_extensions import override
from random import shuffle
from torch import Tensor
from model.utils.ImageList import ImageList


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
                 image_size: Tuple[int, int] = (800, 1330),
                 mask_value: int = 255):

        if ignore_index is None:
            # 一般target中255都是忽略的地方（黑色背景）
            self.ignore_index = [255]
        else:
            self.ignore_index = ignore_index if 255 in ignore_index else ignore_index + [255]
        self.root = root
        self.mask_value = mask_value
        self._check_path_exists(root)
        self.is_filter = False
        self.image_size = image_size
        if not transform:
            self.transform = self._init_image_transform()
        else:
            self.transform = transform
        if not target_transform:
            self.target_transform = self._init_target_transform()
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

    def _init_target_transform(self) -> Callable:
        """把一些需要忽略掉的label对应的target换成255"""

        def process_image(image: Image) -> Image:
            image = np.array(image)
            mask = np.isin(image, self.ignore_index)
            image[mask] = self.mask_value
            return Image.fromarray(image)

        return process_image

    def resize_image_aspect_ratio(self,img:Image)->Image:
        # 原始图片尺寸
        # wd
        min_size = (self.image_size[0], self.image_size[0])
        max_size = (self.image_size[1], self.image_size[0])
        original_width, original_height = img.size

        # 计算纵横比
        aspect_ratio = original_width / original_height

        # 根据纵横比重新计算新的目标尺寸
        if aspect_ratio >= 1:  # 宽图片
            new_width = max_size[0]
            new_height = int(new_width / aspect_ratio)
            if new_height > max_size[1]:  # 如果计算的高度超出范围，则需要按照高度来重新计算
                new_height = max_size[1]
                new_width = int(new_height * aspect_ratio)
                if new_width > max_size[0]: new_width = max_size[0]
        else:  # 高图片
            new_height = max_size[1]
            new_width = int(new_height * aspect_ratio)
            # if new_width > max_size[0]:  # 如果计算的宽度超出范围，则需要按照宽度来重新计算
            #     new_width = max_size[0]
            #     new_height = int(new_width / aspect_ratio)
            #     if new_height > max_size[1]: new_height = max_size[1]

        # resize 图片并保持原始纵横比
        resized_img = img.resize((new_width, new_height), Image.BILINEAR)

        return resized_img

    def _init_image_transform(self) -> Callable:
        new_image_size = self.image_size

        def process_image(image: Image, new_image_size: Tuple[int, int] = new_image_size) -> ImageList:
            new_image = self.resize_image_aspect_ratio(image)
            new_image = torch.tensor(np.array(new_image))
            if len(new_image.shape)!= 3:
                new_image = new_image.unsqueeze(-1)
            new_image = new_image.permute(2, 0, 1)

            image_size = new_image.shape[-2:]
            color_image_size = (3,*new_image_size)
            resize_image = torch.zeros(color_image_size, dtype=torch.float)
            resize_image[:, :image_size[0], :image_size[1]] = new_image
            mask = torch.ones(new_image_size, dtype=torch.bool)
            mask[:image_size[0], :image_size[1]] = False
            image_list = ImageList(resize_image, mask, image_size)
            return image_list

        return process_image

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
        single_batch = {}
        single_batch['path'] = (self.images[index][0], self.images[index][1])
        image = Image.open(self.images[index][0]).convert('RGB')
        target = Image.open(self.images[index][1])
        if not self.is_filter:
            if self.need_index_name:
                text_prompt = self._get_text_prompt_from_target(target) + "."
                single_batch["text_prompt"] = text_prompt
            if self.target_transform is not None:
                target = self.target_transform(target)
            if self.transform is not None:
                image = self.transform(image)
                target = self.transform(target)
            single_batch["data"] = (image, target)
            return single_batch
        else:
            single_batch["data"] = (image, target)
            return single_batch

    def __len__(self):
        return len(self.images)


class BaseIncrement(Dataset):
    """
    不要不要不要不要使用dataloader中的shuffle选项，在dataset中以及手动实现了这个功能，因为暂时没有扒干净dataloader中的代码，shuffle
    还算未定义行为，千万不要用，出现错误不负责
    """

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
        self.dataset.target_transform = self._create_target_transform
        self.index = 0
        self.update_flag = False
        self.class_name = [v for k,v in self.dataset.classes.items() if k in self.labels]

    def __strip_ignore(self, labels):
        for i in self.ignore_index:
            while i in labels:
                labels.remove(i)

    def _create_inverted_order(self, mask_value=255):
        # 映射label和索引
        self.inverted_order = {label: self.order.index(label) for label in self.order if label not in self.ignore_index}

    def _create_target_transform(self, img):

        mask_value = self.mask_value
        image_array = np.array(img)
        if self.masking:
            if self.data_masking == "current":
                tmp_labels = self.labels + [mask_value]
            elif self.data_masking == "current+old":
                tmp_labels = self.labels + self.labels_old + [mask_value]
            else:
                raise ValueError(f"masking type:{self.masking} not supported")
            mask = np.isin(image_array, tmp_labels)
            image_array[~mask] = mask_value
        else:
            mask = np.isin(image_array, self.order)
            image_array[~mask] = mask_value
        return Image.fromarray(image_array)

    def __getitem__(self, index):
        if index >= len(self.dataset.images):
            index = index % len(self.dataset.images)
            shuffle(self.dataset.images)
        if self.update_flag:
            self.update_flag = False
            self.index = index
            data = self.dataset[index - self.index]
            text = data["text_prompt"].split(".")
            text_prompt = [i for i in text if i in self.class_name]
            data["text_prompt"] = text_prompt
        else:
            data = self.dataset[index]
            text = data["text_prompt"].split(".")
            text_prompt = [i for i in text if i in self.class_name]
            data["text_prompt"] = text_prompt
        return data

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
        shuffle(self.dataset.images)
        self.class_name = [v for k, v in self.dataset.classes.items() if k in self.labels]
        self.update_flag = True


class BaseEvaluate(BaseIncrement):
    def __init__(self, no_stage_value=224, **kwargs):
        self.no_stage_value = no_stage_value
        super().__init__(**kwargs)

    @override
    def _create_target_transform(self, img):

        mask_value = self.mask_value
        image_array = np.array(img)
        if self.masking:
            if self.data_masking == "current":
                tmp_labels = self.labels + [mask_value]
            elif self.data_masking == "current+old":
                tmp_labels = self.labels + self.labels_old + [mask_value]
            else:
                raise ValueError(f"masking type:{self.masking} not supported")
            mask = np.isin(image_array, tmp_labels)
            image_array[~mask] = mask_value
        else:
            mask = np.isin(image_array, self.order)
            image_array[~mask] = mask_value
        return image_array
