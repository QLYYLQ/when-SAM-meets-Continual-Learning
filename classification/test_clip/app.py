import torch
from transformers import CLIPProcessor, CLIPModel
import loralib as lora
import gradio as gr
from PIL import Image
import os

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义模型名称和路径
model_name = "openai/clip-vit-base-patch32"
lora_weights_path = r"D:\project\CSS_Filter\test_clip\lora_pottedplant_2classification.pt"


model_path = r"C:\Users\10781\.cache\huggingface\hub\models--openai--clip-vit-base-patch32\snapshots"
clip_model = CLIPModel.from_pretrained(model_name)

clip_model.to(device)
clip_model.eval()

# # 加载 LoRA 权重
# lora_weights = torch.load(lora_weights_path, map_location=device)
#
# # 将 LoRA 权重加载到模型中
# def load_lora_weights(model, lora_weights):
#     own_state = model.state_dict()
#     for name, param in lora_weights.items():
#         if name in own_state:
#             own_state[name].copy_(param)
# load_lora_weights(clip_model, lora_weights)

# 加载处理器
processor = CLIPProcessor.from_pretrained(model_name)

def predict(image, text):
    if image is None:
        return "请上传一张图像。"
    if text.strip() == "":
        return "请输入文本。"
    print(text)
    # 将图像和文本处理为模型所需的格式
    with torch.no_grad():
        text_inputs = processor(text=[text,"black dessert"], return_tensors="pt", padding=True, do_rescale=False)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        text_embeddings = clip_model.get_text_features(**text_inputs)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        image_inputs = processor(images=image, return_tensors="pt", do_rescale=False)
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
        image_embeddings = clip_model.get_image_features(**image_inputs)
        logits = image_embeddings @ text_embeddings.t()
        logits_norm = logits / logits.norm(dim=-1, keepdim=True)


    # 输出结果
    return {
        "logits_norm": logits.tolist()

    }

# 定义输入组件
image_input = gr.Image(type="pil", label="上传图像")
text_input = gr.Textbox(lines=2, placeholder="在此输入文本", label="输入文本")

# 定义输出组件
output = gr.JSON(label="模型输出")

# 创建 Gradio 界面
app = gr.Interface(
    fn=predict,
    inputs=[image_input, text_input],
    outputs=output,
    title="CLIP 模型评估",
    description="上传一张图像并输入文本，模型将输出 logits 值。",
    theme="default"
)

# 启动应用
app.launch()

# #----
# import gradio as gr
# #该函数有3个输入参数和2个输出参数
# def greet(name, is_morning, temperature):
#     salutation = "Good morning" if is_morning else "Good evening"
#     greeting = f"{salutation} {name}. It is {temperature} degrees today"
#     celsius = (temperature - 32) * 5 / 9
#     return greeting, round(celsius, 2)
# demo = gr.Interface(
#     fn=greet,
#     #按照处理程序设置输入组件
#     inputs=["text", "checkbox", gr.Slider(0, 100)],
#     #按照处理程序设置输出组件
#     outputs=["text", "number"],
# )
# demo.launch()



