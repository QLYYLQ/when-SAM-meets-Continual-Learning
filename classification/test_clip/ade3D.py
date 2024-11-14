import numpy as np
import json
import umap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import colorsys
from adjustText import adjust_text  # For automatic text position adjustment

file_path = r"D:\project\CSS_Filter\test_clip\ade.json"

with open(file_path, 'r', encoding='ISO-8859-1') as fo:
    llm_tag_des = json.load(fo)

# Assuming build_openset_llm_label_embedding returns:
# openset_label_embedding: torch.Tensor of shape [150, 512]
# openset_categories: list of length 150 (category names)
from CSS_Filter.ram.utils import build_openset_llm_label_embedding

# Build label embeddings and category list
openset_label_embedding, openset_categories = build_openset_llm_label_embedding(llm_tag_des)

# Convert embeddings from torch.Tensor to numpy array
openset_label_embedding = openset_label_embedding.cpu().numpy()  # Shape [150, 512]

# Use UMAP to reduce embeddings to 3D
reducer = umap.UMAP(n_components=3, random_state=42)
embedding_3d = reducer.fit_transform(openset_label_embedding)  # Shape [150, 3]

# Prepare light pastel colors
num_categories = len(openset_categories)
# Generate evenly distributed hues
hues = np.linspace(0, 1, num_categories, endpoint=False)
# Generate light pastel colors (saturation set to 0.3, lightness to 0.9)
colors = [colorsys.hsv_to_rgb(h, 0.3, 0.9) for h in hues]

# Plot 3D embeddings and label categories
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')

scatter_handles = []
texts = []
for i, (x, y, z) in enumerate(embedding_3d):
    scatter = ax.scatter(x, y, z, color=colors[i], edgecolors='k', s=100)
    scatter_handles.append(scatter)
    # Store text objects for later adjustment
    texts.append(ax.text(x, y, z, openset_categories[i], fontsize=8))

# Adjust text to avoid overlap (Note: adjust_text may not work well in 3D)
adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray'))

ax.set_title('3D UMAP Visualization of Label Embeddings', fontsize=16)
ax.set_xlabel('UMAP First Dimension', fontsize=12)
ax.set_ylabel('UMAP Second Dimension', fontsize=12)
ax.set_zlabel('UMAP Third Dimension', fontsize=12)
ax.grid(True)
plt.tight_layout()
plt.show()

# Write category names and coordinates to file
output_file = r"D:\project\CSS_Filter\test_clip\categories_coordinates_3d.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    for category, (x, y, z) in zip(openset_categories, embedding_3d):
        f.write(f"{category}\t{x:.6f}\t{y:.6f}\t{z:.6f}\n")