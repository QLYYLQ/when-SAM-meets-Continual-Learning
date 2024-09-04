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
    config = _modify_training(config)
    config = _modify_dataset_setting(config)
    config = _modify_increment_setting(config)
    return config


def _modify_training(config):
    training_set = config.training
    classes_number = int(config.dataset_setting.class_number)
    copied_config = deepcopy(config)
    for task, task_setting in training_set.items():
        design = [int(x) for x in task_setting.design.split('-')]
        if sum(design) != classes_number:
            raise ValueError("please check the design. total number is not equal to class number")
        if task_setting.index_order == "none":
            index_order = {}
            stage_number = {x: y for x, y in enumerate(design)}
            label_list = [x for x in range(classes_number)]
            for key, number in stage_number.items():
                index_order[key], label_list = _random_select_and_remove(label_list, number)
            copied_config.training[task].index_order = Munch.fromDict(index_order)
    return copied_config


def _modify_dataset_setting(config):
    classes_path = dataset_config_root_path.joinpath(config.dataset_setting.classes)
    # 修改default的classes name路径
    if config.dataset_setting.classes == 'default':
        classes_path = dataset_config_root_path.joinpath('classes', f"{config.name}_classes.pkl")
    if not osp.exists(classes_path):
        raise FileNotFoundError("please check the file, classes_name is missing")
    with open(classes_path, 'rb') as f:
        classes = pickle.load(f)
    config.dataset_setting.classes = classes
    dataset_root = dataset_root_path.joinpath(config.dataset_setting.root)
    if not osp.exists(dataset_root):
        raise FileNotFoundError(f"please check the file, dataset root is missing. the path is {dataset_root}")
    config.dataset_setting.root = str(dataset_root)

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

def _modify_increment_setting(config):
    pass


def _random_select_and_remove(number_list, n):
    selected = random.sample(number_list, n)
    for num in selected:
        number_list.remove(num)
    return selected, number_list


if __name__ == '__main__':
    config = get_dataset_config('VOC')
    print(config)
    # config = get_dataset_config("ade")
