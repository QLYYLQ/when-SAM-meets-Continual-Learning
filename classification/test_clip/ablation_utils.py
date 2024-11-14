import sys
import os
import tqdm
import torch
import torch.nn as nn
from sympy.codegen import Print
from torch.fx.experimental.unification.unification_tools import getter
from torch.nn import BCELoss, BCEWithLogitsLoss
from torch.nn.functional import mse_loss
from transformers import CLIPProcessor, CLIPModel
import loralib as lora

from CSS_Filter.ram import get_transform

# Set up paths and imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CSS_Filter.css_dataset import load_dataset_from_config, Dataloader, get_dataset_config
from utils import (
    text_prompts_to_tensor,
    load_clip_model,
    BCEWithLogitsLossWithLabelSmoothing, adjust_labels, print_outlier_logits_count, adjust_mean_labels,evaluate_top1_accuracy
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)

config = get_dataset_config("VOC")
config.increment_setting.save_stage_image_path = "default"
dataset, eval = load_dataset_from_config(config, 3, None)
stage_lengths = max(dataset.stage_index_dict.keys())

print("Stage lengths:", stage_lengths + 1)
print(dataset.stage_index_dict[0])  # Stage 0 label
print(dataset.dataset.classes.items())  # dict_items([...])
smoothing = 0.1  # Adjust as needed
loss_fn = BCELoss(reduction="mean")
loss_fn = loss_fn.to(device)
batch_size = 18
num_stages = max(dataset.stage_index_dict.keys())

start = int(input("Please input the first stage: "))
K = int(input("How many epochs?"))

# 定义计算 top1 准确率的函数

for current_stage in range(start, num_stages + 1):
    # 更新数据集和数据加载器
    clip_model = load_clip_model(model_name, with_lora=True)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, clip_model.parameters()), lr=1e-5)

    dataset.update_stage(current_stage)

    # 准备当前阶段的文本提示
    stage_text = dataset.class_name+['Black desert']
    print("123",stage_text)
    num_stage_labels = len(stage_text)

    print(num_stage_labels)
    print(len(stage_text))

    # 计算文本嵌入，并使用 detach()
    print(stage_text)
    text_inputs = processor(text=stage_text, return_tensors="pt", padding=True, do_rescale=False)
    print(text_inputs)

    text_inputs = text_inputs.to(device)
    print(text_inputs)

    text_embeddings = clip_model.clip_model.get_text_features(**text_inputs)
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    text_embeddings = text_embeddings.detach().to(device)
    class_dict = dataset.dataset.classes
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
    # 训练循环
    for k in range(0, K):
        dataset.update_stage(current_stage)
        print(f"Starting Stage {current_stage}_epoch{k}")
        dataloader = Dataloader(dataset, batch_size=batch_size)
        pbar = tqdm.tqdm(total=len(dataloader))

        import matplotlib.pyplot as plt  # 在代码开头导入

        for i, batch in enumerate(dataloader):
            images, _, label_prompts, text_prompts = batch["image"], batch["label_index"], [
                [class_dict[int(idx)] for idx in indices] for indices in batch["label_index"]], batch["text_prompt"]
            # 将数据移动到设备
            images = images.to(device)
            labels_tensor = text_prompts_to_tensor(text_prompts, class_dict, num_stage_labels, stage_text[0],True)
            labels = labels_tensor.to(device).float()

            image_inputs = processor(images=images, return_tensors="pt", do_rescale=False)
            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
            image_embeddings = clip_model.clip_model.get_image_features(**image_inputs)
            #image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
            # 计算 logits
            logits = image_embeddings @ text_embeddings.t()
            print("ori",logits)
            logits = logits/logits.norm(dim=-1, keepdim=True)
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
            torch.nn.utils.clip_grad_norm_(clip_model.parameters(), max_norm=1.0)

            optimizer.step()

            # 计算并打印 top1 准确率
            accuracy = evaluate_top1_accuracy(logits, labels, 0.5)
            print(f"Step {global_step}, Top1 Accuracy: {accuracy:.4f}")

            pbar.update(1)

        pbar.close()
        writer.close()
        model_name_safe = model_name.replace("/", "_")
        torch.save(lora.lora_state_dict(clip_model), f"{model_name_safe}_stage_{current_stage}_LoRA.pt")
        print(f"Completed Stage {current_stage}")
