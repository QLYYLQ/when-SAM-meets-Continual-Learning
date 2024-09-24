
import torch
from torch import nn
class GYF(nn.Module):
    def __init__(self):
        self.clip, _ = clip.load("ViT-B/16", device='cuda')

    def create_textpool_embedding(self, textpool: list):
        run_on_gpu = torch.cuda.is_available()

        with torch.no_grad():
            label_embedding = []
            for text in textpool:
                texts = clip.tokenize(text, truncate=True).cuda()
                if run_on_gpu:
                    texts = texts.cuda()
                text_embeddings = self.clip.encode_text(texts)  # 512
                text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
                label_embedding.append(text_embeddings)

            label_embedding = torch.cat(label_embedding, dim=0)
            if run_on_gpu:
                label_embedding = label_embedding.cuda()

        return label_embedding

if __name__ == '__main__':
    model =GYF()
    print(model)