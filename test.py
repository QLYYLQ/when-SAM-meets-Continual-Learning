from segment_anything_hq import sam_model_registry,SamPredictor
model_type="vit_h"
sam = sam_model_registry[model_type](r"/root/autodl-tmp/when-SAM-meets-Continual-Learning/sam_hq_vit_h.pth")
#prompt_encoder_dict= sam.prompt_encoder.state_dict()
# import torch
# torch.save(prompt_encoder_dict,r"./prompt_encoder.pth")
# print("finish_save")



predictor = SamPredictor(sam)
from PIL import Image
image = Image.open(r"./test1.png")
import numpy as np
image = np.array(image)
image_shape = image.shape[0:2]

predictor.set_image(image)
mask,_,_ = predictor.predict()

