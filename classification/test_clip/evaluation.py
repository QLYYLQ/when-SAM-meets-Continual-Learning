import sys
import os
import tqdm
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
import loralib as lora
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Set up paths and imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CSS_Filter.css_dataset import load_dataset_from_config, Dataloader, get_dataset_config
from CSS_Filter.ram import get_transform


def text_prompts_to_tensor(text_prompts, class_dict, num_classes, first_text):
    """
    Convert text prompts to a tensor suitable for evaluation.
    """
    class_name_to_index = {v: k for k, v in class_dict.items() if 1 <= k <= 20}
    first_index = class_name_to_index[first_text]
    batch_size = len(text_prompts)

    labels_tensor = torch.zeros((batch_size, num_classes), dtype=torch.float32)

    for i, labels in enumerate(text_prompts):
        for label in labels:
            if label in class_name_to_index:
                class_idx = class_name_to_index[label]
                labels_tensor[i][class_idx - first_index] = 1.0

    return labels_tensor


def reverse_normalize(tensor, mean, std):
    """
    Reverse normalization to bring images back to [0, 1].
    """
    device = tensor.device
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(device)
    return tensor * std + mean
def load_original(model_name, device, stage):
    clip_model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    clip_model = clip_model.to(device)
    clip_model.eval()  # Set to evaluation mode
    return clip_model

def load_NoLora_model(model_name,device, stage):
    """
    Load the original CLIP model.
    """
    clip_model = CLIPModel.from_pretrained(model_name)
    print(model_name)


    # Load the saved state_dict
    model_name_safe = model_name.replace("/", "_")
    model_path = f"{model_name_safe}_stage_{stage}_NoLoRA.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    clip_model.load_state_dict(state_dict, strict=False)  # strict=False to allow loading LoRA parameters

    # Freeze the text encoder as during training
    for param in clip_model.text_model.parameters():
        param.requires_grad = False

    clip_model = clip_model.to(device)
    clip_model.eval()  # Set to evaluation mode
    return clip_model
def load_trained_model(model_name, device, stage):
    """
    Load the trained CLIP model along with LoRA weights for a specific stage.
    """
    clip_model = CLIPModel.from_pretrained(model_name)
    print(model_name)
    # Apply LoRA to the image encoder's attention layers with the same rank used during training
    for name, module in clip_model.vision_model.named_modules():
        if isinstance(module, nn.MultiheadAttention) or 'SelfAttention' in name:
            lora.inject_lora(module, r=32)

    # Load the saved state_dict
    model_name_safe = model_name.replace("/", "_")
    model_path = f"{model_name_safe}_stage_{stage}.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    clip_model.load_state_dict(state_dict, strict=False)  # strict=False to allow loading LoRA parameters

    # Freeze the text encoder as during training
    for param in clip_model.text_model.parameters():
        param.requires_grad = False

    clip_model = clip_model.to(device)
    clip_model.eval()  # Set to evaluation mode
    return clip_model


def evaluate_model(model, processor, dataloader, device, class_dict, num_classes, text_embeddings, stage_text):
    """
    Evaluate the model on the evaluation dataset and compute metrics.
    """
    all_preds = []
    all_labels = []
    num_stage_labels = len(stage_text)

    with torch.no_grad():
        pbar = tqdm.tqdm(total=len(dataloader), desc="Evaluating")
        for batch in dataloader:
            images, labels, label_prompts, text_prompts = batch["image"], batch["label_index"], \
                [[class_dict[int(idx)] for idx in indices] for indices in batch["label_index"]], \
                batch["text_prompt"]

            images = images.to(device)
            labels_tensor = text_prompts_to_tensor(text_prompts, class_dict, num_stage_labels, stage_text[0])
            labels = labels_tensor.to(device).float()

            # Reverse normalize images
            #images = reverse_normalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            # Preprocess images and compute image embeddings
            #image_inputs = processor(images=images, return_tensors="pt")
            image_inputs = processor(
                images=images,
                return_tensors="pt",
                do_resize=False,
                do_center_crop=True,
                do_rescale=False,
                do_normalize=False
            )

            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
            image_embeddings = model.get_image_features(**image_inputs)
            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True) #bachsize, 512
            print(image_embeddings)
            # Compute logits
            logits = image_embeddings @ text_embeddings.T

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            # Binarize predictions with a threshold (e.g., 0.5)
            preds = (probs > 0.55).float()

            # Move tensors to CPU and convert to numpy
            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels)

            pbar.update(1)
        pbar.close()

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    report = classification_report(all_labels, all_preds, target_names=class_dict.values())

    print(f"Accuracy for current stage: {accuracy:.4f}")
    print(f"F1 Score for current stage: {f1:.4f}")
    print("Classification Report:")
    print(report)


import numpy as np


def main():
    # Configuration
    stage_to_evaluate = int(input("Please input the stage to evaluate: "))
    config = get_dataset_config("VOC")
    config.increment_setting.save_stage_image_path = "default"
    _, eval_dataset = load_dataset_from_config(config, 1, None)  # Assuming 'eval_dataset' is the second returned value
    eval_dataset.update_stage(stage_to_evaluate)
    eval_dataset.dataset.transform = get_transform()
    stage_lengths = max(eval_dataset.stage_index_dict.keys())
    print("Stage lengths:", stage_lengths)
    print(eval_dataset.stage_index_dict[stage_to_evaluate])  # Stage 0 label
    print(eval_dataset.dataset.classes.items())  # dict_items([...])

    # Initialize CLIP model and processor
    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    cache_dir = os.getenv("TORCH_HOME", os.path.join(str(os.path.expanduser("~")), ".cache"))

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    model = load_NoLora_model(model_name, device, stage_to_evaluate)

    # Prepare evaluation data loader
    batch_size = 8
    dataloader = Dataloader(eval_dataset, batch_size=batch_size)

    # Prepare text embeddings
    stage_text = eval_dataset.class_name
    num_stage_labels = len(stage_text)
    class_dict = eval_dataset.dataset.classes

    text_inputs = processor(text=stage_text, return_tensors="pt", padding=True)
    text_inputs = text_inputs.to(device)
    text_embeddings = model.get_text_features(**text_inputs)
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    text_embeddings = text_embeddings.detach().to(device)

    # Evaluate the model
    evaluate_model(model, processor, dataloader, device, class_dict, num_stage_labels, text_embeddings, stage_text)


if __name__ == "__main__":
    main()
