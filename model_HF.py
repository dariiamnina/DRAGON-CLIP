import torch
import torch.nn as nn
from transformers import CLIPModel
import numpy as np
from SelfAttention_v2 import SelfAttentionLayer

class DRAGONCLIP(nn.Module):
    def __init__(self, clip_model_name="laion/CLIP-ViT-B-32-laion2B-s34B-b79K"):#"openai/clip-vit-large-patch16"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.self_attention = SelfAttentionLayer(512, 1024)

    def forward(self, x, y, inputs_dragon):
        images, text = x, y
        dragon_emb = inputs_dragon

        outputs = self.clip(input_ids=text, pixel_values=images)
        image_features, text_features = outputs.image_embeds, outputs.text_embeds

        # Combine text features with dragon_emb using the attention mechanism
        #input_data = torch.stack([text_features, dragon_emb])
        embed_clip_text = self.self_attention([text_features, dragon_emb])  # Apply self-attention
        embed_clip_text = torch.squeeze(embed_clip_text, -1)

        # Normalize embeddings
        per_image_norm = image_features.norm(dim=-1, keepdim=True)
        per_text_norm = embed_clip_text.norm(dim=-1, keepdim=True)
        image_features = image_features / per_image_norm
        embed_clip_text = embed_clip_text / per_text_norm

        # Compute logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ embed_clip_text.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text


class CLIP_Linear(nn.Module):
    def __init__(self, clip_model_name="laion/CLIP-ViT-B-32-laion2B-s34B-b79K"):#"openai/clip-vit-large-patch16"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x, y):
        images, text = x, y

        outputs = self.clip(input_ids=text, pixel_values=images)
        image_features, text_features = outputs.image_embeds, outputs.text_embeds

        embed_clip_text = text_features

        # Normalize embeddings
        per_image_norm = image_features.norm(dim=-1, keepdim=True)
        per_text_norm = embed_clip_text.norm(dim=-1, keepdim=True)
        image_features = image_features / per_image_norm
        embed_clip_text = embed_clip_text / per_text_norm

        # Compute logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ embed_clip_text.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text


class DRAGONCLIP_debug(nn.Module):
    def __init__(self, clip_model_name="laion/CLIP-ViT-B-32-laion2B-s34B-b79K"):#"openai/clip-vit-large-patch16"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.linear1 = torch.nn.Linear(512, 512)
        self.linear2 = torch.nn.Linear(1536, 512)
        #self.linear3 = torch.nn.Linear(1024, 512)
        self.self_attention = SelfAttentionLayer(512, 1024)

    def forward(self, x, y, dragon_emb):
        images, text = x, y

        outputs = self.clip(input_ids=text, pixel_values=images)
        image_features, text_features = outputs.image_embeds, outputs.text_embeds

        embed_clip_text = torch.cat([text_features, dragon_emb], axis=1)

        image_features = self.linear1(image_features)
        #embed_clip_text = self.linear2(embed_clip_text)
        embed_clip_text_weighted = self.self_attention(embed_clip_text)
        #grads = torch.autograd.grad(outputs=embed_clip_text_weighted, inputs=embed_clip_text, grad_outputs=torch.ones_like(embed_clip_text_weighted))

        # Normalize embeddings
        per_image_norm = image_features.norm(dim=-1, keepdim=True)
        per_text_norm = embed_clip_text_weighted.norm(dim=-1, keepdim=True)
        image_features = image_features / per_image_norm
        embed_clip_text_weighted = embed_clip_text_weighted / per_text_norm

        # Compute logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ embed_clip_text_weighted.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text