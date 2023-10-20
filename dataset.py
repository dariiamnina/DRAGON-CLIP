import io
import pickle
import shutil
import numpy as np
import os
import torch
import random
from PIL import Image


def get_one_hot_vector(class_id, num_classes):
    one_hot_vector = torch.zeros(num_classes)
    one_hot_vector[class_id] = 1
    return one_hot_vector


def split_dataset(files_list, setting, truncate_count):
    train_ratio, val_ratio = (7, 3)
    indices = []
    random.seed(10)
    if setting == 'train':
        for i in range(train_ratio):
            indices += list(range(i, len(files_list), 10))
    if setting == 'val':
        offset = train_ratio
        for i in range(offset, offset + val_ratio):
            indices += list(range(i, len(files_list), 10))
    
    if setting == 'full':
        indices = list(range(0, len(files_list)))

    indices = [indices[idx] for idx in random.sample(range(0, len(indices)), min(len(indices), truncate_count))]
    indices = sorted(indices)
    return [files_list[i] for i in indices]


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, setting='train', overfit=False, truncate=-1, transform=None, dragon=False, dragon_root='DRAGON_train_embeddings/'):
        self.transform = transform
        self.root_dir = root_dir
        self.overfit = overfit
        self.setting = setting
        self.dragon = dragon
        self.dragon_root = dragon_root
        self.truncate = truncate
        self.filenames, self.images, self.textprompts, self.labels, self.dragon_filenames = self.get_data(self.root_dir)
        


    def read_image(self, file_path):
        img = Image.open(file_path)
        img = img.resize((224, 224))
        img = np.asarray(img)
        return img


    def get_data(self, root_dir):
        paths_dict = dict()
        idx_intervals = []

        overfit_num_samples = 16
        
        with open(root_dir + 'train' + '_CLIP_list_of_filenames.pickle', 'rb') as fi:
            filenames = pickle.load(fi)
        with open(root_dir + 'train' + '_CLIP_list_of_images.pickle', 'rb') as fi:
            images = pickle.load(fi)
        with open(root_dir + 'train' + '_CLIP_tokens.pickle', 'rb') as fi:
            textprompts = pickle.load(fi)
        with open(root_dir + 'train' + '_CLIP_list_of_labels.pickle', 'rb') as fi:
            labels = pickle.load(fi)

        random.seed(10)
        indices = [i for i in range(0, len(filenames))]
        random.shuffle(indices)

        if self.truncate:
            indices = indices[:min(len(indices), self.truncate)]

        train_val_split = 0.7
        if self.setting == 'train':
            indices = indices[:int(len(indices)*train_val_split)]
        elif self.setting == 'val':
            indices = indices[int(len(indices)*train_val_split):]
        
        if self.overfit:
            indices = indices[0:overfit_num_samples]

        filenames = [filenames[i] for i in indices]
        images = [images[i] for i in indices]
        textprompts = [textprompts[i] for i in indices]
        labels = [labels[i] for i in indices]

        if self.dragon:
            dragon_filenames = [str(i) + '.pickle' for i in indices]
        else:
            dragon_filenames = None
        
        return filenames, images, textprompts, labels, dragon_filenames


    def __len__(self):
        return len(self.filenames)


    def get_length(self):
        return self.__len__()


    def __getitem__(self, idx):
        class_id = self.labels[idx]
        image = np.array(self.images[idx])
        image =  Image.fromarray(image)
        image = image.resize((224, 224))
        image = np.asarray(image)
        textprompt = self.textprompts[idx]
        textprompt = textprompt.type(torch.int64)

        class_one_hot = get_one_hot_vector(class_id, num_classes=200)

        if self.transform is None:
            image_tensor = torch.Tensor(image)
            class_one_hot = torch.Tensor(class_one_hot)
        else:
            if image.shape[-1] == 4:
                image = image[:, :, :-1] # remove alpha channel, its all 255
            if len(image.shape) == 2:
                image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            image = Image.fromarray(image)
            image_tensor = self.transform(image)
            class_one_hot = torch.Tensor(class_one_hot)

        
        if self.dragon:
            with open(self.dragon_root + self.dragon_filenames[idx], 'rb') as fi:
                dragon_embedding = pickle.load(fi)
            return image_tensor, class_one_hot, class_id, textprompt, dragon_embedding
        else:
            return image_tensor, class_one_hot, class_id, textprompt

        

