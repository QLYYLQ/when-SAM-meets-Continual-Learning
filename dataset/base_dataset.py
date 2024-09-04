import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


# 这里存放的是dataset的模板，继承这个模板实现相应的功能就可以了

class BaseSegmentation(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, need_index_name=True, classes=None):
        self.root = root
        self._check_path_exists(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.need_index_name = need_index_name
        if need_index_name and classes is None:
            raise ValueError("you have to specify the classes when need index_name")
        self.classes = classes
        self.classes[255]="ignore"
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
                x = line.split(",")
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
        target_text = [self.classes[x] for x in unique_values]
        text_prompt = ".".join(target_text)
        return text_prompt

    def __getitem__(self, index):
        image = Image.open(self.images[index][0]).convert('RGB')
        target = Image.open(self.images[index][1])
        if self.transform is not None:
            image, target = self.transform(image, target)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.need_index_name:
            text_prompt = self._get_text_prompt_from_target(target)
            return {"data": (image, target), "data_path": (self.images[index][0],self.images[index][1]), "text_prompt": text_prompt}
        return {"data": (image, target), "data_path": (self.images[index][0], self.images[index][1])}


if __name__ == '__main__':
    path = r"F:\Code_Field\Python_Code\Pycharm_Code\dataset\my_dataset\data\PascalVOC12"
