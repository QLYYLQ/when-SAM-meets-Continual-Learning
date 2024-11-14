import os
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from CSS_Filter.css_dataset import load_dataset_from_config, get_dataset_config, Dataloader
from CSS_Filter.ram import get_transform
from torchvision.utils import save_image, make_grid
from torchvision import transforms

def crop_and_save(images, i, k):
    """
    裁剪每个batch的中心区域，并将它们拼接成一张图片保存。

    Args:
        images: 一个batch的图片，形状为 (batch_size, channels, height, width)，这里假设是 (9, 3, 384, 384)。
    """
    batch_size, channels, height, width = images.shape
    crop_size = 128  # 裁剪大小为384

    # 逆归一化
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inv_normalize = transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std])
    images = inv_normalize(images)
    save_path = Path("D:\\project\\CSS_Filter\\nearOOD_image")
    if not save_path.exists():
        save_path.mkdir()
    cropped_images = []
    j=0
    for img in images:
        save_image(img, f"D:\\project\\CSS_Filter\\nearOOD_image\\ori{k}_{j}.png")
        j+=1
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size

        cropped_img = img[:, top:bottom, left:right]
        cropped_images.append(cropped_img)

    # 拼接成 3x3 的网格
    grid = make_grid(cropped_images, nrow=3,padding=0)
    # 将网格图片保存
    save_path = Path("D:\\project\\CSS_Filter\\nearOOD_image")
    if not save_path.exists():
        save_path.mkdir()
    save_file = save_path / f"sheep{i}.png"
    save_image(grid, save_file)

# Set device (CUDA or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get dataset configuration
config = get_dataset_config("VOC")
config.increment_setting.save_stage_image_path = "default"

# User-specified task number
task_number = [5]
s = int(input("Step into specific task number:"))

# Load the dataset
for i in task_number:
    if s != 0:
        i = s
    train, eval = load_dataset_from_config(config, i, None)
    train.dataset.transform = get_transform(image_size=384)

    start = int(input("Enter the starting stage: "))
    K = 1
    print(f"Classes: {train.dataset.classes}")
    num_stages = max(eval.stage_index_dict.keys())
    k=0
    for current_stage in range(start, num_stages + 1):
        # Update dataset and DataLoader for each stage
        train.update_stage(current_stage)
        stage_text = train.class_name
        num_stage_labels = len(stage_text)
        print(f"Stage {current_stage} - Labels: {num_stage_labels}")

        # Prepare DataLoader
        dataloader = Dataloader(train, batch_size=9)
        pbar = tqdm(total=len(dataloader))

        # Training loop (this is just data processing, no model or loss)
        for i, batch in enumerate(dataloader):
            images, _, label_prompts, text_prompts = batch["image"], batch["label_index"], batch["label_index"], batch[
                "text_prompt"]

            # Move data to device (GPU/CPU)

            k+=1
            images = images.to(device)
            crop_and_save(images, i,k)
            pbar.update(1)
        break
        pbar.close()
