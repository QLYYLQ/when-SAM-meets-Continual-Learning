import torch
import clip
from torch import nn
def generate_masks_with_special_tokens_and_transfer_map(tokenized, special_tokens_list=[101, 102, 1012, 1029]):
    """Generate attention mask between each pair of special tokens
    Args:
        input_ids (torch.Tensor): input ids. Shape: [bs, num_token]
        special_tokens_mask (list): special tokens mask.
    Returns:
        torch.Tensor: attention mask between each special tokens.
    """
    input_ids = tokenized
    bs, num_token = input_ids.shape
    # special_tokens_mask: bs, num_token. 1 for special tokens. 0 for normal tokens
    special_tokens_mask = torch.zeros((bs, num_token), device=input_ids.device).bool()
    for special_token in special_tokens_list:
        special_tokens_mask |= input_ids == special_token

    # idxs: each row is a list of indices of special tokens
    idxs = torch.nonzero(special_tokens_mask)

    # generate attention mask and positional ids
    attention_mask = (
        torch.eye(num_token, device=input_ids.device).bool().unsqueeze(0).repeat(bs, 1, 1)
    )
    position_ids = torch.zeros((bs, num_token), device=input_ids.device)
    cate_to_token_mask_list = [[] for _ in range(bs)]
    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if (col == 0) or (col == num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1 : col + 1, previous_col + 1 : col + 1] = True
            position_ids[row, previous_col + 1 : col + 1] = torch.arange(
                0, col - previous_col, device=input_ids.device
            )
            c2t_maski = torch.zeros((num_token), device=input_ids.device).bool()
            c2t_maski[previous_col + 1 : col] = True
            cate_to_token_mask_list[row].append(c2t_maski)
        previous_col = col

    # cate_to_token_mask_list = [
    #     torch.stack(cate_to_token_mask_listi, dim=0)
    #     for cate_to_token_mask_listi in cate_to_token_mask_list
    # ]

    # # padding mask
    # padding_mask = tokenized['attention_mask']
    # attention_mask = attention_mask & padding_mask.unsqueeze(1).bool() & padding_mask.unsqueeze(2).bool()

    return attention_mask, position_ids.to(torch.long)


class clip_encoder(nn.Module):
    def __init__(self):
        super(clip_encoder,self).__init__()
        self.clip, _ = clip.load("ViT-B/16", device='cuda')

    def forward(self, textpool: list):
        run_on_gpu = torch.cuda.is_available()

        with torch.no_grad():
            label_embedding = []
            for text in textpool:
                texts = clip.tokenize(text, truncate=True).cuda()
                if run_on_gpu:
                    texts = texts.cuda()
                text_embeddings = self.clip.encode_text(texts)  # 512
                text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
                (text_self_attention_masks,position_ids,) = generate_masks_with_special_tokens_and_transfer_map(text_embeddings)
                text_dict = {
                            "encoded_text": text_embeddings,  # bs, 195, d_model
                            "text_token_mask": torch.ones(text_embeddings.shape,dtype = torch.bool),  # bs, 195
                            "position_ids": position_ids,  # bs, 195
                            "text_self_attention_masks": text_self_attention_masks,  # bs, 195,195
                            }
                label_embedding.append(text_dict)

            # label_embedding = torch.cat(label_embedding, dim=0)
            # if run_on_gpu:
                # label_embedding = label_embedding.cuda()
            return label_embedding
    

def build_text_encoder(encoder_name):
    if encoder_name != "CLIP":
        raise ValueError("only have CLIP text encoder")
    return clip_encoder()

    
if __name__ == "__main__":
    model = clip_encoder()
    token = model.create_textpool_embedding(["tv monitor","sofa","desk"])