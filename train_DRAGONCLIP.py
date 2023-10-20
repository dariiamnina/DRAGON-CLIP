import clip
import copy
import torch
import time
import os
from datetime import datetime
from model import DRAGONCLIP_debug
from dataset import ImageDataset
from torch.utils.tensorboard import SummaryWriter

# from transformers import CLIPTextModel, CLIPTokenizer
# tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
# text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
print(device)
torch.cuda.empty_cache()

writer = SummaryWriter()

_, preprocess = clip.load("ViT-B/32", device=device, jit=False)
model = DRAGONCLIP_debug()

loss_img = torch.nn.CrossEntropyLoss()
loss_txt = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.9,0.98), eps=1e-6, weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    # for p in model.parameters(): 
    #     p.data = p.data.float() 
    #     p.grad.data = p.grad.data.float() 
    with open('dragonclip_params.txt', 'w') as fo:
        for name, param in model.named_parameters():
            fo.write(f'{name} {param.size()}\n')
            param.data = param.data.float()
            if name == 'logit_scale' or name == 'clip.logit_scale':
                continue
            param.grad.data = param.grad.data.float()



def train_model(model, loss_img, loss_txt, optimizer, epochs=25,
                save_path='checkpoints/CLIP_classifier_v1.pth', load_path=None):

    if load_path and os.path.exists(load_path):
        model.load_state_dict(torch.load(load_path))
        print(f'Loaded model from {load_path}')
    model.to(device)
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs - 1}')
            print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            if phase == 'train':
                epoch_start = time.time()

            # Iterate over data.
            for images, _, class_ids, texts, dragon_emb in dataloaders[phase]:
                optimizer.zero_grad()
                images = images.to(device)
                texts = texts.to(device)
                dragon_emb = dragon_emb.to(device)
                logits_per_image, logits_per_text = model(images, texts, dragon_emb)

                ground_truth = torch.arange(len(images),dtype=torch.long,device=device)

                total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth))/2

                total_loss.backward()
                ###convert_models_to_fp32(model)
                optimizer.step()
                ###clip.model.convert_weights(model)

                # statistics
                running_loss += total_loss.item() * images.size(0)
                running_corrects += (torch.sum(torch.max(logits_per_image, 1).indices == ground_truth.data) + \
                torch.sum(torch.max(logits_per_text, 1).indices == ground_truth.data))/2

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if epoch % 10 == 0:
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                writer.add_scalar(f'loss/{phase}', epoch_loss, epoch)
                writer.add_scalar(f'accuracy/{phase}',  epoch_acc, epoch)
                if phase == 'train':
                    print(f'Epoch time: {time.time() - epoch_start}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Saving the model')
                torch.save(model.state_dict(), save_path)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    writer.close()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


root_dir = 'data/train/'

datasets = {
    'train': ImageDataset(root_dir=root_dir, setting='train', overfit=False, truncate=400, transform=preprocess, dragon=True),
    'val': ImageDataset(root_dir=root_dir, setting='val', overfit=False, truncate=400, transform=preprocess, dragon=True)
    }
dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=64, shuffle=True, num_workers=0) \
               for x in ['train', 'val']}
dataset_sizes = {'train': datasets['train'].get_length(), \
                 'val': datasets['val'].get_length()
                 }

print(dataset_sizes)

dt = datetime.now()

trained_model = train_model(model, loss_img, loss_txt, optimizer, epochs=1000, save_path= f'checkpoints/DRAGONCLIP_{dt}.pth', load_path='checkpoints/DRAGONCLIP_2023-10-01 14:38:04.154119.pth')
