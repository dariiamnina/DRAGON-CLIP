import clip
import numpy as np
import torch
from torch import nn
from SelfAttention import SelfAttention

class DRAGONCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip, self.preprocess = clip.load("ViT-B/32", device='cuda:0', jit=False) #Must set jit=False for training
        self.attention = SelfAttention(input_dim=1) # sequence of 1024 by 1
        self.fuse = nn.Linear(1024+512, 512)
        

    def forward(self, x, y, z):
        clip.model.convert_weights(self.clip)
        image, text, embed_dragon = x, y, z

        embed_clip_image = self.clip.encode_image(image)
        embed_clip_text = self.clip.encode_text(text)
        
        embed_dragon = torch.unsqueeze(embed_dragon, -1)
        embed_dragon_weighted = self.attention(embed_dragon)
        embed_dragon_weighted = torch.squeeze(embed_dragon_weighted, -1)
        
        # Fuse embeddings
        embed_text_fused = self.fuse(torch.cat([embed_clip_text, embed_dragon_weighted], dim=1))
        return embed_clip_image, embed_text_fused
        

class DRAGONCLIP_debug(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip, self.preprocess = clip.load("ViT-B/32", device='cuda:0', jit=False) #Must set jit=False for training
        self.attention = SelfAttention(input_dim=1) # sequence of 1024+512 by 1
        self.fuse = nn.Linear(1536, 512)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


    def forward(self, x, y, z):
        image, text, dragon_emb = x, y, z
        
        embed_clip_image = self.clip.encode_image(image)
        embed_clip_text = self.clip.encode_text(text)

        concat_emb = torch.cat([embed_clip_text, dragon_emb], dim=1)
        concat_emb = torch.unsqueeze(concat_emb, -1)
        concat_emb_weighted  = self.attention(concat_emb)
        concat_emb = torch.squeeze(concat_emb_weighted, -1)

        embed_clip_text = self.fuse(concat_emb)

        per_image_norm = embed_clip_image.norm(dim=-1, keepdim=True)
        per_text_norm = embed_clip_text.norm(dim=-1, keepdim=True)
        embed_clip_image = embed_clip_image / per_image_norm
        embed_clip_text = embed_clip_text / per_text_norm
        embed_clip_text = embed_clip_text.type(torch.float16)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * embed_clip_image @ embed_clip_text.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text


class CLIP_Linear(nn.Module):
    def __init__(self):
        super().__init__()
        # Fusion layers
        self.fuse1 = nn.Linear(1024*2, 512) 
        self.fuse2 = nn.Linear(512, 256)

    def forward(self, x):
        # Fuse embeddings
        fuse1 = self.fuse1(x)
        fuse2 = self.fuse2(F.relu(fuse1))
        
        return fuse2


class CLIP_Linear_v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip, self.preprocess = clip.load("ViT-B/32", device='cuda:0', jit=False) #Must set jit=False for training
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


    def forward(self, x, y):
        image, text = x, y
        
        embed_clip_image = self.clip.encode_image(image)
        embed_clip_text = self.clip.encode_text(text)

        per_image_norm = embed_clip_image.norm(dim=-1, keepdim=True)
        per_text_norm = embed_clip_text.norm(dim=-1, keepdim=True)
        embed_clip_image = embed_clip_image / per_image_norm
        embed_clip_text = embed_clip_text / per_text_norm
        embed_clip_text = embed_clip_text.type(torch.float16)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * embed_clip_image @ embed_clip_text.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text