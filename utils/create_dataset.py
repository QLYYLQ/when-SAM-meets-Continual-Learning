from pathlib import Path
from os import path as osp
from utils import auto_init
from dataset import dataset_entrypoints,is_dataset_registered


config_path = Path(__file__).resolve().parent.parent.parent.joinpath('config')

def load_dataset_from_config(config,logger):
    pass

def create_task_dataset(config,task_number:int,logger):
    task_number = "task_"+str(task_number)
    task = config.training[task_number]
    # design = task.design
    index_order = task.index_order
    dataset = {}
    dataset_name = config.increment_setting.segmentation_dataset_name
    if not is_dataset_registered(dataset_name):
        raise ValueError(f"in your config, the dataset:{dataset_name} is not registered, please check out or use "
                         f"default instead")
    labels_old = []
    for key,value in index_order.items():
        stage_number = key
        labels = value
        # 创建数据集
        dataset_class = dataset_entrypoints(dataset_name)
        init_dict = auto_init(dataset_class,config.increment_setting)
        if config.increment_setting.segmentation_config is "default":
            init_dict["segmentation_config"] = config.dataset_setting
        if config.increment_setting.save_stage_image_list_path is "default":
            # init_dict["save_stage_image_list_path"]
            root_save_stage_image_list_path = config_path.joinpath("dataset","save_stage_image_list",config.name)
            path_name = root_save_stage_image_list_path.joinpath(str(labels)+".txt")
            if osp.exists(path_name):
                init_dict["save_stage_image_list_path"] = path_name
