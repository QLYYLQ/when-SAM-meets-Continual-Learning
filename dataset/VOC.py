import os

import torch.utils.data as data
from .register import register_training_dataset, register_validation_dataset
from dataset.base_dataset import BaseSplit,BaseIncrement
from typing_extensions import override



@register_training_dataset
class Split(BaseSplit):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    @override
    def _get_path(self):
        # train 和 train_aug 文件一样
        return os.path.join(self.root, "splits","train.txt")




@register_training_dataset
class Increment(BaseIncrement):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)




@register_validation_dataset
class Validation(data.Dataset):
    def __init__(
            self,
            root,
            classes = None,
            transform=None,
            target_transform=None,
            order=None,
            labels=None,
            labels_old=None,
            idxs_path=None,
            save_path=None,
            masking=True,
            overlap=True,
            data_masking="current",
            **kwargs
    ):
        """感觉测试数据中的实现不需要考虑是否overlap，后续包装中可以通过传入的labels调整dataset为disjoint或者overlap"""
        self.root = root
        self.dataset = Split(root, need_index_name=True, transform=transform, classes=classes, target_transform=target_transform)
        self.order = order
        # 感觉这两个参数目前用不到
        self.labels = labels
        self.labels_old = labels_old
        #
        self.save_path = save_path

