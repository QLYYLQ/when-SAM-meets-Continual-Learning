import sys
import os
import tqdm
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型名称
model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CSS_Filter.css_dataset import load_dataset_from_config, Dataloader, get_dataset_config

def text_prompts_to_tensor(text_prompts, class_dict, num_classes, first_text):
    class_name_to_index = {v: k for k, v in class_dict.items() if 1 <= k <= 20}
    first_index = class_name_to_index[first_text]
    batch_size = len(text_prompts)
    labels_tensor = torch.zeros((batch_size, num_classes), dtype=torch.float32)
    for i, labels in enumerate(text_prompts):
        for label in labels:
            if label in class_name_to_index:
                class_idx = class_name_to_index[label]
                labels_tensor[i][class_idx - first_index] = 0.35
    return labels_tensor

def load_clip_model():
    clip_model = CLIPModel.from_pretrained(model_name).to(device)
    for param in clip_model.text_model.parameters():
        param.requires_grad = False
    return CLIPWithLearnableTemperature(clip_model)

class CLIPWithLearnableTemperature(nn.Module):
    def __init__(self, clip_model):
        super(CLIPWithLearnableTemperature, self).__init__()
        self.clip_model = clip_model
        self.temperature = nn.Parameter(torch.tensor(0.3))

    def forward(self, image_inputs, text_embeddings):
        image_embeddings = self.clip_model.get_image_features(**image_inputs)
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        print(image_embeddings)
        logits = (image_embeddings @ text_embeddings.T) / self.temperature
        return logits

def train_one_stage(current_stage, K, dataset, processor, loss_fn, batch_size, model_name):
    clip_model = load_clip_model()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, clip_model.parameters()), lr=1e-6)
    dataset.update_stage(current_stage)
    stage_text = dataset.class_name
    num_stage_labels = len(stage_text)
    text_inputs = processor(text=stage_text, return_tensors="pt", padding=True, do_rescale=False).to(device)
    text_embeddings = clip_model.clip_model.get_text_features(**text_inputs)
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    text_embeddings = text_embeddings.detach()
    class_dict = dataset.dataset.classes
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
    for k in range(K):
        dataloader = Dataloader(dataset, batch_size=batch_size)
        pbar = tqdm.tqdm(total=len(dataloader))
        for i, batch in enumerate(dataloader):
            images = batch["image"].to(device)
            labels_tensor = text_prompts_to_tensor(batch["text_prompt"], class_dict, num_stage_labels, stage_text[0])
            labels = labels_tensor.to(device)
            image_inputs = processor(images=images, return_tensors="pt", do_rescale=False)
            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
            logits = clip_model(image_inputs, text_embeddings)
            print("logits", logits)
            loss = loss_fn(logits, labels)
            print("loss",loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('Loss/train', loss.item(), i)
            writer.add_scalar('Temperature', clip_model.temperature.item(), i)
            pbar.update(1)
        pbar.close()
    writer.close()
    model_name_safe = model_name.replace("/", "_")
    torch.save(clip_model.state_dict(), f"{model_name_safe}_stage_{current_stage}_NoLoRA.pt")
    print(f"Completed Stage {current_stage}")

def main():
    config = get_dataset_config("VOC")
    config.increment_setting.save_stage_image_path = "default"
    dataset, eval = load_dataset_from_config(config, 1, None)
    smoothing = 0.1
    loss_fn = BCEWithLogitsLossWithLabelSmoothing(smoothing=smoothing).to(device)
    batch_size = 16
    num_stages = max(dataset.stage_index_dict.keys())
    start = int(input("请输入开始的阶段（整数）："))
    K = int(input("请输入每个阶段的训练轮数（整数）："))
    for current_stage in range(start, num_stages + 1):
        train_one_stage(current_stage, K, dataset, processor, loss_fn, batch_size, model_name)

class BCEWithLogitsLossWithLabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super(BCEWithLogitsLossWithLabelSmoothing, self).__init__()
        self.smoothing = smoothing
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        loss = self.bce_loss(logits, targets)
        return loss

if __name__ == "__main__":
    main()
