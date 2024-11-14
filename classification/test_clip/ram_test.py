import torch
from torch import nn
from CSS_Filter.css_dataset import load_dataset_from_config, get_dataset_config, Dataloader
from CSS_Filter.ram.models import ram_plus
from CSS_Filter.ram import inference_ram_openset as inference
from CSS_Filter.ram import get_transform
import numpy as np
from CSS_Filter.ram.utils import build_openset_llm_label_embedding
from torch import nn
import json

from CSS_Filter.ram.models import ram_plus

# 初始化设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 设定模型的标记嵌入等参数
file_path = r"D:\project\ram\recognize-anything\datasets\tag_descriptions.json"
with open(file_path, 'r', encoding='ISO-8859-1') as fo:
    llm_tag_des = json.load(fo)

    openset_label_embedding, openset_categories = build_openset_llm_label_embedding(llm_tag_des)

openset_label_embedding = openset_label_embedding.cpu().numpy()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import json


# 降维
tsne = TSNE(n_components=3, random_state=42)
embeddings_3d = tsne.fit_transform(openset_label_embedding)



# 假设embeddings_3d和categories已经定义

# 创建一个新的3D图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 为20个类别定义颜色
colors = plt.cm.rainbow(np.linspace(0, 1, 20))

# 对每个类别进行pooling并绘制
for i, category in enumerate(openset_categories):
    # 获取当前类别的所有点
    category_points = embeddings_3d[i * 50:(i + 1) * 50]

    # 对这些点进行pooling（取平均值）
    pooled_point = np.mean(category_points, axis=0)

    # 绘制pooled点
    ax.scatter(pooled_point[0], pooled_point[1], pooled_point[2],
               c=[colors[i]], label=category, s=100, alpha=1)

# 设置图表标题和标签
ax.set_title("Pooled Category Embeddings")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# 添加图例
ax.legend()

plt.show()

