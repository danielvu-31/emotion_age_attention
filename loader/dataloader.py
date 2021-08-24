import os
# import cv2
# import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from .auto_augment import ImageNetPolicy


class FaceDataset(Dataset):
    def __init__(self,
                img_root,
                label_path,
                task=["age", "emotion"],
                inp_img_size=224,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                phase='train'):
        super().__init__()
        self.img_root = img_root
        self.phase = phase
        self.task = task
        col_used = [f"{name}_label" for name in task]
        col_used.extend(['item_id', 'face_id'])
        label_df = pd.read_csv(label_path, usecols=col_used)

        if self.phase == 0: # 0: train, 1: val
            self.transform = transforms.Compose(
                [transforms.Resize(inp_img_size),
                ImageNetPolicy(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)]
                )
        else:
            self.transform = transforms.Compose(
                [transforms.Resize(inp_img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)]
                )
        
        self.item_id_list = label_df['item_id'].to_list()
        self.face_id_list = label_df['face_id'].to_list()
        self.labels, self.number_per_class = {}, {}
        for name in task:
            self.labels['name'] = label_df[f"{name}_label"].to_list()
            self.number_per_class[name] = {}
            for value in set(self.labels['name']):
                if value != -1:
                    self.number_per_class[name][str(value)] = len(label_df[f"{name}_label"==value].to_list())
        
    def __len__(self):
        return len(self.item_id_list)
    
    def __getitem__(self, index):
        image_id = self.item_id_list[index]
        face_id = self.face_id_list[index]
        try:
            image_path = os.path.join(self.img_root, f'{image_id}_{face_id}.jpg')
            image = Image.open(image_path)
            image = self.transform(image)
        except Exception as exception:
            print('Exception:', exception)
            return self.__getitem__(index + 1)
        
        for key in self.labels.keys():
            if key == 'age':
                age_label = self.labels[key][index]
            elif key == 'emotion':
                emotion_label = self.labels[key][index]

        return image_id, face_id, image, age_label, emotion_label