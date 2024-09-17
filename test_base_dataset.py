# import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).parent.parent.absolute()))
import numpy as np
from dataset.base_dataset import BaseSplit
from typing_extensions import override
import os
from utils import get_dataset_config, auto_init

# from dataset.base_dataset import BaseIncrement
# from dataset.VOC import Segmentation
# class TestIncrement(BaseIncrement):
#     def __init__(self,**kwargs):
#         super().__init__(**kwargs)
#
#
# dataset_config = get_dataset_config("VOC")
# print(dataset_config)
# # print(dataset_config)
# init_dict = auto_init(TestIncrement,dataset_config.increment_setting)
# print(init_dict)
# init_dict1 = auto_init(Segmentation,dataset_config.dataset_setting)
# print(init_dict1)(
# init_dict["segmentation_config"]=init_dict1
# init_dict["labels"]=dataset_config.training.task_1.index_order[1]
# init_dict["labels_old"]=dataset_config.training.task_1.index_order[0]
#
# dataset = TestIncrement(**init_dict)
# # dataset = Segmentation(**init_dict)
# for i in dataset:
#     print(i["text_prompt"])


# config = get_dataset_config("ADE")
# from dataset import dataset_entrypoints
#
# DATASET=dataset_entrypoints("ADE.Segmentation")
# init_dict = auto_init(DATASET,config.dataset_setting)
#
# dataset = DATASET(**init_dict)
# for i in dataset:
#     print(i["text_prompt"])
#     print(i["scene"])
#
#
# test_list = [13,5]
# print(str(test_list))
from torch.utils.data import DataLoader
from utils.create_dataset import load_dataset_from_config
from utils import get_dataset_config
from dataset.dataloader import Dataloader

config = get_dataset_config("VOC")
print(config)
dataset = load_dataset_from_config(config,1,None)
dataloader = Dataloader(dataset,batch_size=15)
i=0
for img,label,text in dataloader:
    print([s.tensor.shape for s in img])
    i+=1
    # print(text)
    if i >= 90:
        dataset.update_stage(1)