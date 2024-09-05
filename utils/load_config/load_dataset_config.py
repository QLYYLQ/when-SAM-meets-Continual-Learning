import os.path as osp
import random
from pathlib import Path
from .base_load import get_config
from munch import Munch
import pickle
from copy import deepcopy

dataset_config_root_path = Path(__file__).resolve().parent.parent.parent.joinpath('config', 'dataset')
dataset_root_path = Path(__file__).resolve().parent.parent.parent.joinpath("data")


def get_dataset_config(dataset_name):
    config_path = dataset_config_root_path.joinpath(dataset_name + '.yaml')
    config = get_config(config_path)
    template_config = _load_template_config()
    _check_config(config, template_config)
    config = _modify_dataset_config(config)
    return config


def _load_template_config():
    config_path = dataset_config_root_path.joinpath('template.yaml')
    if not osp.exists(config_path):
        raise FileNotFoundError("please check the file, template is missing")
    config = get_config(config_path)
    return config


def _modify_dataset_config(config):
    """change default setting in config"""
    config.training = _modify_training(config.training, config.class_number)
    config.dataset_setting = _modify_dataset_setting(config.dataset_setting, config.name)
    config.increment_setting = _modify_increment_setting(config.increment_setting, config.name)
    return config


def _modify_training(config, class_number):
    copied_config = deepcopy(config)
    for task, task_setting in config.items():
        design = [int(x) for x in task_setting.design.split('-')]
        if sum(design) != class_number:
            raise ValueError("please check the design. total number is not equal to class number")

        label_list = [x for x in range(class_number)]
        stage_number = {x: y for x, y in enumerate(design)}
        index_order = {}
        if task_setting.index_order == "random":
            for key, number in stage_number.items():
                index_order[key], label_list = _random_select_and_remove(label_list, number)
        elif task_setting.index_order == "default":
            last_number = 0
            for key, number in stage_number.items():
                number += last_number
                index_order[key] = label_list[last_number:number]
                last_number = number
        copied_config[task].index_order = Munch.fromDict(index_order)
    return copied_config


def _modify_dataset_setting(config, dataset_name):
    classes_path = dataset_config_root_path.joinpath(config.classes)
    # 修改default的classes name路径
    if config.classes == 'default':
        classes_path = dataset_config_root_path.joinpath('classes', f"{dataset_name}_classes.pkl")
    if not osp.exists(classes_path):
        raise FileNotFoundError("please check the file, classes_name is missing")
    with open(classes_path, 'rb') as f:
        classes = pickle.load(f)
    config.classes = classes
    dataset_root = dataset_root_path.joinpath(config.root)
    if not osp.exists(dataset_root):
        raise FileNotFoundError(f"please check the file, dataset root is missing. the path is {dataset_root}")
    config.root = str(dataset_root)

    return config


def _modify_increment_setting(config, dataset_name):
    if config.segmentation_dataset_name == "default":
        config.segmentation_dataset_name = dataset_name + ".Segmentation"
    if config.save_stage_image_list_path == "default":
        #这里在后续加载任务中实现，不是在这里实现
        pass
    return config


def _check_config(config, template_config):
    for setting, _ in template_config.items():
        if not hasattr(config, setting):
            raise AttributeError(f"please check the config, {setting} is missing")
    if not hasattr(config.training, "task_1"):
        raise AttributeError("must design task_1 at first")

    for task, task_setting in config.training.items():
        for setting, _ in template_config.training.task_1.items():
            if not hasattr(task_setting, setting):
                raise AttributeError(f"please check the config, in {task}, the setting: {setting} is missing")
    for task, task_setting in config.evaluate.items():
        for setting, _ in template_config.evaluate.task_1.items():
            if not hasattr(task_setting, setting):
                raise AttributeError(f"please check the config, in {task}, the setting: {setting} is missing")
    for task, _ in config.evaluate.items():
        if not hasattr(config.training, task):
            raise AttributeError(f"please check the config, evaluate and training should have same task {task}")
    for setting, _ in template_config.dataset_setting.items():
        if not hasattr(config.dataset_setting, setting):
            raise AttributeError(f"please check the config, you should have dataset setting: {setting}")
    for setting, _ in template_config.increment_setting.items():
        if not hasattr(config.dataset_setting, setting):
            raise AttributeError(f"please check the config, you should have increment dataset setting: {setting}")


def _random_select_and_remove(number_list, n):
    selected = random.sample(number_list, n)
    for num in selected:
        number_list.remove(num)
    return selected, number_list


if __name__ == '__main__':
    config = get_dataset_config('VOC')
    print(config)
    # config = get_dataset_config("ade")
