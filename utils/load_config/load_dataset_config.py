import os.path as osp
from munch import Munch
from pathlib import Path
from base_load import get_config

dataset_config_root_path = Path(__file__).resolve().parent.parent.parent.joinpath('config', 'dataset')


def get_dataset_config(dataset_name):
    config_path = dataset_config_root_path.joinpath(dataset_name + '.yaml')
    config = get_config(config_path)
    template_config = _load_template_config()
    _check_config(config, template_config)
    config = _modify_dataset_config(config, template_config)
    return config


def _load_template_config():
    config_path = dataset_config_root_path.joinpath('template.yaml')
    if not osp.exists(config_path):
        raise FileNotFoundError("please check the file, template is missing")
    config = get_config(config_path)
    return config


def _modify_dataset_config(config, template_config):
    """change default setting in config"""
    training_set = config.training
    for task, task_setting in training_set.items:

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


if __name__ == '__main__':
    config = get_dataset_config('VOC')
    print(config)
