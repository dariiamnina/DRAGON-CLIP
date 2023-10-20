import clip
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from dataset import ImageDataset

model, preprocess = clip.load("ViT-B/32")

root_dir = 'data/train/'

dataset = ImageDataset(root_dir=root_dir, setting='train', overfit=True, transform=preprocess, truncate=1e6)
val_dataset = ImageDataset(root_dir=root_dir,setting='val', overfit=True, transform=preprocess, truncate=1e6)


train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
print(dataset.get_length())
image_tensor, class_one_hot, class_id, textprompt = next(iter(train_dataloader))
print(f"Feature batch shape: {image_tensor.size()}")
print(f"Labels batch shape: {class_one_hot.size()}")
print(f"Class ids batch shape: {class_id.size()}")
print(f"Textprompts batch shape: {len(textprompt)}")
for batch in train_dataloader:
    print(batch[0].shape)
    pass