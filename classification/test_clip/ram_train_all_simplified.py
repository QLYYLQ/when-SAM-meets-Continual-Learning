import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import loralib as lora
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
from utils import  AsymmetricLoss
from CSS_Filter.ram import get_transform

from CSS_Filter.ram.utils import build_openset_llm_label_embedding
from torch import nn
import json
print("gyf RUNINGG")
# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = get_transform(image_size=384)
model = ram_plus(pretrained=r"D:\project\CSS_Filter\ram\pretrained\ram_plus_swin_large_14m.pth",
                 image_size=384,
                 vit='swin_l')
#model = lora.Lora(model, r=16)
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
    model = model.to(device)
#lora.mark_only_lora_as_trainable(model)
task_number=[5]
loss_fn = AsymmetricLoss(gamma_neg=4,gamma_pos=1,clip=0.05)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


# 启用所有参数的梯度
for name, param in model.named_parameters():
    param.requires_grad = True

# 创建优化器，包含所有可训练参数
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

for i in task_number:
    config = get_dataset_config("VOC")
    config.increment_setting.save_stage_image_path = "default"
    train, eval = load_dataset_from_config(config, i, None)
    train.dataset.transform = get_transform(image_size=384)
    start = int(input("Enter the starting stage: "))
    K = 1
    print(train.dataset.classes)
    num_stages = max(eval.stage_index_dict.keys())

    for current_stage in range(start, num_stages + 1):
        # 更新数据集和数据加载器
        # 准备当前阶段的文本提示
        stage_text = train.class_name
        print("123",stage_text)
        num_stage_labels = len(stage_text)

        print(num_stage_labels)
        print(len(stage_text))

        # 计算文本嵌入，并使用 detach()
        print(stage_text)


        class_dict = train.dataset.classes
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()
        # 训练循环
        for k in range(0, K):
            train.update_stage(current_stage)
            print(f"Starting Stage {current_stage}_epoch{k}")
            dataloader = Dataloader(train, batch_size=16)
            pbar = tqdm(total=len(dataloader))
            from utils import text_prompts_to_tensor, evaluate_top1_accuracy
            import matplotlib.pyplot as plt  # 在代码开头导入

            for i, batch in enumerate(dataloader):
                images, _, label_prompts, text_prompts = batch["image"], batch["label_index"], batch["label_index"], batch["text_prompt"]
                # 将数据移动到设备
                images = images.to(device)
                labels_tensor = text_prompts_to_tensor(text_prompts, class_dict, num_stage_labels, stage_text[0],False)
                labels = labels_tensor.to(device).float()

                logits = model(images)
                print("ori",logits)
                # 打印异常值数量
                #print_outlier_logits_count(logits, 3)
                # 打印 logits 和 labels
                print(logits, labels)
                # 计算损失
                loss = loss_fn(logits, labels)
                print(loss)
                global_step = k * len(dataloader) + i

                # 记录损失和温度
                writer.add_scalar('Loss/train', loss.item(), global_step)

                # 使用 Matplotlib 绘制 logits 和 labels 的图形
                logits_np = logits[0].cpu().detach().numpy() - 0.5
                labels_np = labels[0].cpu().detach().numpy()
                x = range(len(logits_np))  # x轴：0到17
                fig = plt.figure()
                plt.plot(x, logits_np, label='Logits', marker='o')
                plt.plot(x, labels_np, label='Labels', marker='x')
                plt.title(f'Logits and Labels at Step {global_step}')
                plt.xlabel('类别索引')
                plt.ylabel('数值')
                plt.legend()

                # 将图形添加到 TensorBoard
                writer.add_figure('Logits_vs_Labels', fig, global_step)
                plt.close(fig)  # 关闭图形以释放内存

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 计算并打印 top1 准确率
                accuracy = evaluate_top1_accuracy(logits, labels, 0.5)
                print(f"Step {global_step}, Top1 Accuracy: {accuracy:.4f}")

                pbar.update(1)

            pbar.close()
            writer.close()

