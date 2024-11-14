import sys
import os
import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CSS_Filter.css_dataset import load_dataset_from_config, Dataloader, get_dataset_config

# 加载数据集配置
config = get_dataset_config("VOC")
config.increment_setting.save_stage_image_path = "default"
dataset, eval = load_dataset_from_config(config, 1, None)

stage_lengths = max(dataset.stage_index_dict.keys())

print("Stage lengths:", stage_lengths)
print(dataset.stage_index_dict[0])  # Stage 0 label
print(dataset.dataset.classes.items())  # dict_items([...])

# 定义变换
transform = get_transform(image_size=384)
dataset.dataset.transform = transform

# 设置 Dataloader
batch_size = 8
dataloader = Dataloader(dataset, batch_size=batch_size)

i = 0
current_stage = 0
class_mapping = dataset.dataset.classes

# 遍历 stage
for current_stage in dataset.stage_index_dict.keys():
    print(f"Processing stage {current_stage}/{stage_lengths}")

    # 重置 dataloader 来处理当前 stage
    stage_length = len(dataset)
    dataset.update_stage(current_stage)
    print(dataset.class_name)

    dataloader = Dataloader(dataset, batch_size=batch_size)

    # 初始化 tqdm 进度条
    pbar = tqdm.tqdm(total=len(dataloader), desc=f"Stage {current_stage}/{stage_lengths}")

    for batch in dataloader:
        data = batch["image"]
        label_index = batch["label_index"]
        label_names = [[class_mapping[int(idx)] for idx in indices] for indices in label_index]
        text_prompt = batch["text_prompt"]

        print(data.shape)
        print(label_names)
        print(text_prompt)

        i += batch_size

        # 更新进度条
        pbar.update(1)

        if i >= stage_length:
            print(f"Completed stage {current_stage + 1}")
            current_stage += 1  # 进入下一个 stage
            i = 0  # 重置计数器
            break  # 跳出当前循环，开始下一个 stage

    # 关闭进度条
    pbar.close()
