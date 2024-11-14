import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
config_path = Path(__file__).resolve().parent.joinpath('config')

import torch

classes = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike', 'person',
    'potted plant', 'sheep', 'sofa', 'train', 'tv monitor'
]

from CSS_Filter.css_dataset import load_dataset_from_config, get_dataset_config, Dataloader
from CSS_Filter.ram.models import ram_plus
from CSS_Filter.ram import inference_ram_openset as inference
from CSS_Filter.ram import get_transform

from CSS_Filter.ram.utils import build_openset_llm_label_embedding
from torch import nn
import json

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = get_transform(image_size=384)

model = ram_plus(pretrained=r"D:\project\CSS_Filter\ram\pretrained\ram_plus_swin_large_14m.pth",
                 image_size=384,
                 vit='swin_l')

print('Building tag embedding:')
file_path = r"D:\project\ram\recognize-anything\datasets\tag_descriptions.json"

with open(file_path, 'r', encoding='ISO-8859-1') as fo:  # 或者尝试 'GBK'
    llm_tag_des = json.load(fo)

    openset_label_embedding, openset_categories = build_openset_llm_label_embedding(llm_tag_des)

    model.tag_list = np.array(openset_categories)

    model.label_embed = nn.Parameter(openset_label_embedding.float())

    model.num_class = len(openset_categories)
    # the threshold for unseen categories is often lower
    model.class_threshold = torch.ones(model.num_class) * 0.6
    #######

    model.eval()

    model = model.to(device)

task_number=[3,4]


for i in task_number:
    config = get_dataset_config("VOC")
    config.increment_setting.save_stage_image_path = "default"
    _, eval = load_dataset_from_config(config, i, None)
    eval.dataset.transform = get_transform(image_size=384)
    start = 0
    K = 1
    num_stages = max(eval.stage_index_dict.keys())
    for current_stage in range(start, num_stages + 1):
        eval.update_stage(current_stage)
        print(f"Starting Stage {current_stage}")

        # 设置根路径，并确保路径存在
        root_dir = f"D:\\project\\VOC\\task_{i}\\val"
        os.makedirs(root_dir, exist_ok=True)
        txt_path = os.path.join(root_dir, f"{current_stage}.txt")

        # 打开 txt 文件，准备写入
        with open(txt_path, 'w', encoding='utf-8') as f:
            pbar = tqdm(total=len(eval))
            class_dict = eval.dataset.classes
            missing_labels_count = {}
            for idx, batch in enumerate(eval):
                #print(batch)
                path = batch["path"]
                labels = [class_dict[label_idx] for label_idx in batch['label_index']]
                images = torch.unsqueeze(batch["data"][0], 0)
                #print(path)
                #print(labels)
                images = images.to(device)
                res = inference(images, model).tolist()
                missing_labels = ["potted plant", "person", "tv monitor", "bottle", "dinning table"]
                for label in labels:
                    if label in missing_labels and label not in res:
                        #print(label)
                        res.append(label)
                print("res", res)
                print("labels", labels)

                # 将 path[0], path[1], res 写入 txt 文件
                line = f"{path[0]}, {path[1]}, {res}, {labels}\n"
                f.write(line)

                pbar.update(1)
            pbar.close()

        pbar.close()
        print(f"Completed Stage {current_stage}")

