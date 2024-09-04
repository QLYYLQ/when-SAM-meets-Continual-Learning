import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
import numpy as np
from dataset.base_dataset import BaseSegmentation
from typing_extensions import override
import os
from utils import get_dataset_config, auto_init


class TestSegmentation(BaseSegmentation):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.type = "test"

    @override
    def _get_path(self):
        return os.path.join(self.root, "splits","train.txt")

    @override
    def _load_data_path_to_list(self, path):
        images = []
        with open(path, 'r') as f:
            for line in f:
                x = line.strip().split(" ")
                images.append((os.path.join(self.root, x[0][1:]), os.path.join(self.root, x[1][1:])))
        return images

    @override
    def _get_text_prompt_from_target(self, target):
        unique_values = np.unique(np.array(target).flatten())
        target_text = [self.classes[x] for x in unique_values if x not in [0,255]]
        text_prompt = ".".join(target_text)
        return text_prompt

dataset_config = get_dataset_config("VOC")
print(dataset_config)
init_dict = auto_init(TestSegmentation,dataset_config.dataset_setting)
print(init_dict)
dataset = TestSegmentation(**init_dict)
for i in dataset:
    print(i["text_prompt"])