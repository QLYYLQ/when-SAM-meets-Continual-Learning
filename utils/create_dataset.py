from pathlib import Path
import os
from utils import auto_init, get_dataset_config
from dataset import dataset_entrypoints, is_dataset_registered, filter_images, save_list_from_filter

config_path = Path(__file__).resolve().parent.parent.joinpath('config')


def load_dataset_from_config(config, task_number, logger):
    dataset_class = create_task_dataset(config, task_number, logger)
    return dataset_class


def create_task_dataset(config, task_number: int, logger):
    """创建选择任务的数据集"""
    task_number = "task_" + str(task_number)
    task = config.training[task_number]
    # design = task.design
    index_order = task.index_order
    config.dataset_setting["ignore_index"] = task.ignore_index
    increment_dataset_name = config.increment_setting.increment_dataset_name
    split_dataset_name = config.increment_setting.split_dataset_name
    if not is_dataset_registered(increment_dataset_name):
        raise ValueError(
            f"in your config, the dataset:{increment_dataset_name} is not registered, please check out or use "
            f"default instead")
    increment_dataset_class = dataset_entrypoints(increment_dataset_name)
    split_dataset_class = dataset_entrypoints(split_dataset_name)
    init_dict = auto_init(increment_dataset_class, config.increment_setting)
    init_split_dict = auto_init(split_dataset_class, config.dataset_setting)
    if config.increment_setting.split_config == "default":
        init_dict["split_config"] = init_split_dict
    if config.increment_setting.save_stage_image_path == "None":
        dataset = split_dataset_class(**init_dict["split_config"])
        stage_path_dict = init_task_path_from_config(config, dataset, task_number)
    else:
        root_save_stage_image_list_path = config_path.joinpath("dataset", "saved_task_txt", config.name,task_number)
        stage_path_dict = {stage: str(root_save_stage_image_list_path.joinpath(str(stage) + ".txt")) for stage in
                            index_order.keys()}
    init_dict["stage_path_dict"] = stage_path_dict
    init_dict["stage_index_dict"] = index_order
    increment = increment_dataset_class(**init_dict)
    return increment


def init_task_path_from_config(config, dataset, task_number):
    root_save_stage_image_list_path = config_path.joinpath("dataset", "saved_task_txt", config.name)
    index_order = config.training[task_number].index_order
    if config.increment_setting.save_stage_image_path == "None":
        os.makedirs(str(root_save_stage_image_list_path.joinpath(task_number)), exist_ok=True)
        # 调用迭代分割函数
        create_image_path_from_task(config, dataset, task_number)
    stage_image_path = {stage: str(root_save_stage_image_list_path.joinpath(task_number,str(stage) + ".txt")) for stage in
                        index_order.keys()}
    return stage_image_path


def create_image_path_from_task(config, dataset, task_number: str):
    task = config.training[task_number]
    index_order = dict(task.index_order)
    root_save_stage_image_list_path = config_path.joinpath("dataset", "saved_task_txt", config.name, task_number)
    stage_path = {stage: str(root_save_stage_image_list_path.joinpath(str(stage) + ".txt")) for stage in
                  index_order.keys()}
    for stage, save_path in stage_path.items():
        if stage == 0:
            create_image_path_from_stage(dataset, index_order[stage], [], save_path,
                                         overlap=config.increment_setting.overlap)
        else:
            create_image_path_from_stage(dataset, index_order[stage], index_order[stage - 1], save_path,
                                         overlap=config.increment_setting.overlap)


def create_image_path_from_stage(dataset, labels, labels_old, save_path, overlap=True):
    image_list_path = filter_images(dataset, labels, labels_old, overlap)
    # root_save_stage_image_list_path = config_path.joinpath("dataset", "save_stage_image_list", config.name)
    save_list_from_filter(image_list_path, save_path)


if __name__ == "__main__":
    config = get_dataset_config("ADE")
    print(config)
    dataset = load_dataset_from_config(config, task_number=1, logger=None)
    from PIL import Image
    import numpy as np
    from matplotlib.colors import hex2color
    def tensor_to_image(tensor,i):
        # 确保tensor是CPU上的
        tensor = tensor.cpu()

        # 将tensor从(C, H, W)转换为(H, W, C)
        tensor = tensor.permute(1, 2, 0)
        tensor = tensor * 255
        # 将tensor转换为numpy数组
        numpy_array = tensor.numpy()

        # 确保值在0-255范围内
        numpy_array = numpy_array.clip(0, 255).astype('uint8')

        # 创建PIL图像
        image = Image.fromarray(numpy_array)
        image.save("F:\\Code_Field\\Python_Code\\Pycharm_Code\\dataset\\my_dataset\\test_pic"+f"\\{i}.png")


    color_map = {
        1: '#FF0000', 2: '#00FF00', 3: '#0000FF', 4: '#FFFF00', 5: '#FF00FF',
        6: '#00FFFF', 7: '#800000', 8: '#008000', 9: '#000080', 10: '#808000',
        11: '#800080', 12: '#008080', 13: '#FFA500', 14: '#A52A2A', 15: '#DEB887',
        16: '#5F9EA0', 17: '#7FFF00', 18: '#D2691E', 19: '#FF7F50', 20: '#6495ED',255:"#000000"
    }
    def label_to_image(label,i1):
        height, width = label.shape[-2],label.shape[-1]
        image = np.zeros((height, width, 3), dtype=np.uint8)
        label = label*255
        for i in range(1, 21):
            mask = (label == i)
            color = np.array(hex2color(color_map[i]))
            image[mask] = (color * 255).astype(np.uint8)
        image = Image.fromarray(image)
        image.save("F:\\Code_Field\\Python_Code\\Pycharm_Code\\dataset\\my_dataset\\test_pic" + f"\\label_{i1}.png")



    for i,batch in enumerate(dataset):
        tensor_to_image(batch["data"][0],i)
        label_to_image(batch["data"][1],i)
        print(batch["text_prompt"])
        if i >5:
            dataset.update_stage(1)
