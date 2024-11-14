import numpy as np
import json
import umap
import matplotlib.pyplot as plt
import colorsys
from adjustText import adjust_text  # 用于自动调整文本位置，避免遮挡

file_path = r"D:\project\CSS_Filter\test_clip\ade.json"

with open(file_path, 'r', encoding='ISO-8859-1') as fo:
    llm_tag_des = json.load(fo)

# 假设 build_openset_llm_label_embedding 返回如下：
# openset_label_embedding: torch.Tensor of shape [150, 512]
# openset_categories: list of length 150（每个类别的名称）
from CSS_Filter.ram.utils import build_openset_llm_label_embedding

# 构建标记嵌入和类别列表
openset_label_embedding, openset_categories = build_openset_llm_label_embedding(llm_tag_des)

# 将嵌入从 torch.Tensor 转换为 numpy 数组
openset_label_embedding = openset_label_embedding.cpu().numpy()  # 形状为 [150, 512]

# 使用 UMAP 将嵌入降维到二维
reducer = umap.UMAP(n_components=2, random_state=42)
embedding_2d = reducer.fit_transform(openset_label_embedding)  # 形状为 [150, 2]

# 准备浅饱和度的颜色
num_categories = len(openset_categories)
# 生成均匀分布的色调
hues = np.linspace(0, 1, num_categories, endpoint=False)
# 生成浅饱和度的颜色（饱和度设为0.3，亮度设为0.9）
colors = [colorsys.hsv_to_rgb(h, 0.3, 0.9) for h in hues]

# 绘制二维嵌入并标注类别
plt.figure(figsize=(15, 10))

scatter_handles = []
texts = []
for i, (x, y) in enumerate(embedding_2d):
    scatter = plt.scatter(x, y, color=colors[i], edgecolors='k', s=100)
    scatter_handles.append(scatter)
    # 将文本对象存储起来，供后续调整
    texts.append(plt.text(x, y, openset_categories[i], fontsize=8))

# 调整文本以避免重叠
adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray'))

plt.title('2D UMAP Visualization of Label Embeddings', fontsize=16)
plt.xlabel('UMAP First Dimension', fontsize=12)
plt.ylabel('UMAP Second Dimension', fontsize=12)
plt.grid(True)
plt.tight_layout()

plt.savefig("umap_visualization.svg", transparent=True)