from .init_from_config import auto_init
from dataset import dataset_entrypoints,is_dataset_registered




def load_dataset_from_config(config):
    pass

def create_task_dataset(config,task_number,logger):
    task = config.training[task_number]
    design = task.design
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
            init_dict["segmentation_config"] = config.dataset_settings



