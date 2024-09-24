from segment_anything_hq import sam_model_registry,SamPredictor
model_type="vit_h"
sam = sam_model_registry[model_type](r"/root/autodl-tmp/when-SAM-meets-Continual-Learning/sam_hq_vit_h.pth")
#prompt_encoder_dict= sam.prompt_encoder.state_dict()
# import torch
# torch.save(prompt_encoder_dict,r"./prompt_encoder.pth")
# print("finish_save")
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


predictor = SamPredictor(sam)
from PIL import Image
image = Image.open(r"./test1.png")
import numpy as np
image = np.array(image)
image_shape = image.shape[0:2]

predictor.set_image(image)
mask,_,_ = predictor.predict()