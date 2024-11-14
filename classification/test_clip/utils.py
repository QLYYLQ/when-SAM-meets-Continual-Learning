# utils.py

import torch
import torch.nn as nn
from sympy import false
from torch.nn import BCELoss
from transformers import CLIPProcessor, CLIPModel
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import loralib as lora
import torch.nn.functional as F

#from CSS_Filter.test_clip.ablation_vis import labels_tensor
def text_prompt_to_all_labels(text_prompt, class_dict):
    num_classes = len(class_dict)
    labels_tensor = torch.zeros(num_classes, dtype=torch.float32)

    for idx, class_name in class_dict.items():
        if class_name in text_prompt:
            labels_tensor[idx - 1] = 1.0  # 索引从0开始，所以要减1

    return labels_tensor
def evaluate_top1_accuracy(logits, labels, threshold=None):
    """
    计算 top-1 准确率。根据 threshold 参数选择计算方法：
    - 如果 threshold 被指定，则使用 softmax 后的概率大于该阈值作为正类判断条件。
    - 如果 threshold 未指定，则使用标准的 argmax 计算。

    参数：
    - logits: 模型输出的 logits，形状为 (batch_size, num_classes)
    - labels: 真实标签，形状为 (batch_size, num_classes)
    - threshold: 进行正类判断的阈值。如果为 None，则使用 argmax 方法。

    返回：
    - accuracy: top-1 准确率
    """
    if threshold is not None:
        # 使用 softmax 获取概率
        probabilities = F.softmax(logits, dim=1)
        #print("asdfadsf", probabilities)
        # 找到每个样本中概率最大的类别索引
        predicted_indices = torch.argmax(probabilities, dim=1)  # shape: (batch_size,)

        # 找到每个样本中真实标签中概率大于阈值的位置
        true_labels = probabilities > threshold

        # 计算预测是否在真实标签中
        correct = 0
        for i in range(len(predicted_indices)):
            if true_labels[i, predicted_indices[i]]:
                correct += 1
    else:
        # 未指定阈值时，使用标准的 argmax 计算
        predicted_indices = torch.argmax(logits, dim=1)
        true_indices = torch.argmax(labels, dim=1)

        # 计算预测是否正确
        correct = (predicted_indices == true_indices).sum().item()

    # 计算准确率
    accuracy = correct / len(logits)
    return accuracy
class BCEWithLogitsLossWithLabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1, pos_weight=None):
        super(BCEWithLogitsLossWithLabelSmoothing, self).__init__()
        self.smoothing = smoothing
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        targets = targets * (1 - self.smoothing) + self.smoothing * (1 - targets)
        loss = self.bce_loss(logits, targets)
        return loss
def text_prompts_to_tensor(text_prompts, class_dict, num_classes, first_text, num_1=False):
    if num_1:
        labels_tensor = torch.zeros((len(text_prompts), 2), dtype=torch.float32)
        labels_tensor[:, 0] = 1.0  # 将第0位置设置为1
        return labels_tensor
    # Create a mapping from class names to indices (1 to 20)
    else:
        class_name_to_index = {v: k for k, v in class_dict.items() if 1 <= k <= 20}
        first_index = class_name_to_index[first_text]
        batch_size = len(text_prompts)

        # Initialize the tensor with zeros
        labels_tensor = torch.zeros((batch_size, num_classes), dtype = torch.float32 )
        #labels_tensor = torch.full((batch_size, num_classes), 0.0, dtype=torch.float32)

        # Populate the tensor
        for i, labels in enumerate(text_prompts):
            for label in labels:
                if label in class_name_to_index:
                    class_idx = class_name_to_index[label]
                    # Subtract 1 because tensor indices start at 0
                    labels_tensor[i][class_idx - first_index] = 1.0
        assert torch.all((labels_tensor >= 0) & (labels_tensor <= 1)), "Labels must be in [0, 1] range."
        return labels_tensor

def print_outlier_logits_count(logits, y):
    mean = torch.mean(logits)
    std_dev = torch.std(logits)
    outliers = (logits > mean + y * std_dev) | (logits < mean - y * std_dev)
    count_outliers = torch.sum(outliers).item()
    print(f"Number of logits deviating more than {y} standard deviations: {count_outliers}")


class CLIPWithVisLoRA(torch.nn.Module):
    def __init__(self, clip_model):
        super(CLIPWithVisLoRA, self).__init__()
        self.clip_model = clip_model
        # Apply LoRA to the parts of the CLIP model you want to train
        self.apply_lora(self.clip_model)

    def apply_lora(self, model):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # 获取父模块和属性名称
                parent_module, attr_name = self.get_parent_module_and_attr_name(model, name)
                # 替换为 LoRA 线性层
                lora_layer = lora.Linear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    r=8,  # LoRA 秩，可以根据需要调整
                    lora_alpha=16,  # LoRA 放大系数
                    lora_dropout=0.1  # LoRA dropout 概率
                )
                # 复制原始权重和偏置
                lora_layer.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    lora_layer.bias = module.bias
                else:
                    lora_layer.bias = None
                # 将 LoRA 层移动到与原始层相同的设备
                lora_layer.to(module.weight.device)
                # 替换模块
                setattr(parent_module, attr_name, lora_layer)

    def get_parent_module_and_attr_name(self, model, module_name):
        # 获取父模块和属性名称
        modules = module_name.split('.')
        parent = model
        for mod in modules[:-1]:
            parent = getattr(parent, mod)
        return parent, modules[-1]

    def forward(self, image_inputs, text_embeddings):
        image_embeddings = self.clip_model.get_image_features(**image_inputs)
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        return image_embeddings @ text_embeddings.T

def load_clip_model(model_name, with_lora=True):
    clip_model = CLIPModel.from_pretrained(model_name).to("cuda")
    if with_lora:
        model = CLIPWithVisLoRA(clip_model)
        # 将模型移动到设备上
        model.to("cuda")
        # 仅训练 LoRA 参数
        lora.mark_only_lora_as_trainable(model)
        return model
    else:
        for param in clip_model.text_model.parameters():
            param.requires_grad = False
        return CLIPWithLearnableTemperature(clip_model)

import torch

def adjust_labels(logits, original_labels, k=250.0, k0=0.12):
    """
    根据 logits 的均值和标准差调整标签。

    参数：
    - logits: 模型输出的 logits，形状为 (batch_size, num_classes)
    - original_labels: 原始标签，形状为 (batch_size, num_classes)，取值为 0 或 1
    - k: 控制标签偏移程度的超参数

    返回：
    - adjusted_labels: 调整后的标签，形状为 (batch_size, num_classes)
    """
    k0 = k0 * k
    with torch.no_grad():
        # 计算每个样本的均值和标准差
        mean = logits.mean(dim=1, keepdim=True)
        std = logits.std(dim=1, keepdim=True)

        # 初始化调整后的标签
        adjusted_labels = torch.zeros_like(original_labels, dtype=logits.dtype)

        # 生成与原始标签形状相同的随机数
        random_vals = torch.rand_like(original_labels, dtype=logits.dtype)

        # 对于标签为 0 的类别，根据随机数选择增加或减少 0.4 * k * std 的扰动
        perturbation = torch.where(random_vals < 0.5, - k0 * std, k0 * std).expand_as(original_labels)
        adjusted_labels[original_labels == 0] = (mean + perturbation)[original_labels == 0]

        # 对于标签为 1 的类别，设置为均值加偏移
        adjusted_labels[original_labels == 1] = (mean + 0.3).expand_as(original_labels)[original_labels == 1]

        # 对调整后的标签进行截断，使其不小于 0
        adjusted_labels = torch.clamp(adjusted_labels, min=0.0)
        adjusted_labels = torch.clamp(adjusted_labels, max=1.0)
    return adjusted_labels , std


def adjust_mean_labels(logits, original_labels, k=250.0, k0=0.12):
    """
    根据 logits 的均值和标准差调整标签。

    参数：
    - logits: 模型输出的 logits，形状为 (batch_size, num_classes)
    - original_labels: 原始标签，形状为 (batch_size, num_classes)，取值为 0 或 1
    - k: 控制标签偏移程度的超参数

    返回：
    - adjusted_labels: 调整后的标签，形状为 (batch_size, num_classes)
    """
    k0 = k0 * k * 0.2

    with torch.no_grad():
        # 找到标签为 0 和 1 的 logits
        logits_0 = logits[original_labels == 0]
        logits_1 = logits[original_labels == 1]

        # 计算分别的均值和标准差
        mean0 = logits_0.mean(dim=0, keepdim=True)
        std0 = logits_0.std(dim=0, keepdim=True)
        mean1 = logits_1.mean(dim=0, keepdim=True)
        std1 = logits_1.std(dim=0, keepdim=True)

        # 初始化调整后的标签
        adjusted_labels = torch.zeros_like(original_labels, dtype=logits.dtype)

        # 对于标签为 0 的类别，根据随机数选择增加或减少 0.4 * k * std 的扰动
        random_vals = torch.rand_like(original_labels, dtype=logits.dtype)
        perturbation = torch.where(random_vals < 0.5, - k0 * std0, 0).expand_as(original_labels)
        adjusted_labels[original_labels == 0] = (mean0 + perturbation)[original_labels == 0]

        # 调整标签为 1 的类别的 logits 根据公式
        condition = mean1[0] - mean0[0] > 0
        adjusted_logits_1 = torch.where(condition, mean1 + 10*(mean1 - mean0) +0.1, mean0+0.2)

        adjusted_labels[original_labels == 1] = adjusted_logits_1.expand_as(logits_1)

        # 对调整后的标签进行截断，使其处于有效范围内
        adjusted_labels = torch.clamp(adjusted_labels, min=0.0, max=1.0)

    return adjusted_labels


class CLIPWithLearnableTemperature(nn.Module):
    def __init__(self, clip_model):
        super(CLIPWithLearnableTemperature, self).__init__()
        self.clip_model = clip_model
        # 初始化温度参数
        #self.temperature = nn.Parameter(torch.tensor(0.3))  # 学习温度参数

    def forward(self, image_inputs, text_embeddings):
        image_embeddings = self.clip_model.get_image_features(**image_inputs)
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)

        logits = (image_embeddings @ text_embeddings.T)
        return logits


def get_loss_function(loss_type="bce", smoothing=0.1, pos_weight=None, device='cpu'):
    """
    获取指定类型的损失函数

    参数:
        loss_type (str): 损失函数类型
        smoothing (float): 标签平滑参数
        pos_weight (Tensor, optional): 正样本权重
        device (torch.device): 设备

    返回:
        nn.Module: 损失函数
    """
    if loss_type == "bce":
        return BCELoss(reduction="sum").to(device)
    elif loss_type == "bce_with_smoothing":
        return BCEWithLogitsLossWithLabelSmoothing(smoothing=smoothing, pos_weight=pos_weight).to(device)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

def get_optimizer(model, lr=2e-6):
    """
    获取优化器

    参数:
        model (nn.Module): 模型
        lr (float): 学习率

    返回:
        torch.optim.Optimizer: 优化器
    """
    return torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

def setup_writer(log_dir="runs"):
    """
    设置 TensorBoard 的 SummaryWriter

    参数:
        log_dir (str): 日志目录

    返回:
        SummaryWriter: TensorBoard writer
    """
    return SummaryWriter(log_dir)

def log_figures(writer, logits, labels, global_step):
    """
    使用 Matplotlib 绘制 logits 和 labels 的图形，并添加到 TensorBoard

    参数:
        writer (SummaryWriter): TensorBoard writer
        logits (Tensor): logits 张量
        labels (Tensor): labels 张量
        global_step (int): 全局步骤
    """
    logits_np = logits.cpu().detach().numpy() - 0.5
    labels_np = labels.cpu().detach().numpy()
    x = range(len(logits_np))  # x轴：类别索引

    fig = plt.figure()
    plt.plot(x, logits_np, label='Logits', marker='o')
    plt.plot(x, labels_np, label='Labels', marker='x')
    plt.title(f'Logits and Labels at Step {global_step}')
    plt.xlabel('类别索引')
    plt.ylabel('数值')
    plt.legend()

    writer.add_figure('Logits_vs_Labels', fig, global_step)
    plt.close(fig)  # 关闭图形以释放内存

def train_stage(stage, K, dataset, clip_model, processor, class_dict, num_stage_labels, stage_text, text_embeddings, loss_fn, optimizer, device, writer, batch_size=8):
    """
    封装每个阶段的训练逻辑

    参数:
        stage (int): 当前阶段
        K (int): 训练轮数
        dataset (Dataset): 数据集
        clip_model (CLIPWithLearnableTemperature): 模型
        processor (CLIPProcessor): CLIP 处理器
        class_dict (dict): 类别字典
        num_stage_labels (int): 当前阶段的标签数量
        stage_text (list): 当前阶段的文本标签
        text_embeddings (Tensor): 文本嵌入
        loss_fn (nn.Module): 损失函数
        optimizer (torch.optim.Optimizer): 优化器
        device (torch.device): 设备
        writer (SummaryWriter): TensorBoard writer
        batch_size (int): 批量大小
    """
    pass
def calculate_accuracy_and_print_results(label_prompts, predictions):
    # 将标签和预测转换为集合以进行比较
    label_set = set(label_prompts[0])  # 只考虑第一个标签（如果有多个）
    prediction_set = set(predictions)

    # 计算缺少的标签
    missing_labels = label_set - prediction_set  # 计算缺少的标签
    # if missing_labels:
    #     print(f"Missing labels: {missing_labels}")

    # 计算正确的标签数量
    correct_predictions = label_set & prediction_set  # 找到正确的预测标签
    accuracy = len(correct_predictions) / len(label_set) if label_set else 0  # 计算正确率

    return accuracy, missing_labels


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()