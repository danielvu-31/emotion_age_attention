import os
import pandas as pd
import copy
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
from base import BaseDataLoader
from .auto_augment import ImageNetPolicy


class FaceDataset(BaseDataLoader):
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
        col_used.extend(['item_id', 'face_id', 'x1', 'y1', 'x2', 'y2'])
        label_df = pd.read_csv(label_path, usecols=col_used)

        self.coordinate_per_image = dict()
        image_id_list = label_df["item_id"].to_list()

        # (x1, y1) - left-upper corner of bounding box
        # (x2, y2) - right-bottom corner of bounding box
        x1_list = label_df["x1"].to_list()
        y1_list = label_df["y1"].to_list()
        x2_list = label_df["x2"].to_list()
        y2_list = label_df["y2"].to_list()

        for id, x1, y1, x2, y2 in  \
            zip(image_id_list, x1_list, y1_list, x2_list, y2_list):
            if id not in self.coordinate_per_image.keys():
                self.coordinate_per_image[id] = []
            self.coordinate_per_image[id].append((x1, y1, x2, y2))

        if self.phase == "train":
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
        
        self.item_id_list = label_df.to_list()
        self.face_id_list = label_df['face_id'].to_list()
        self.labels, self.number_per_class = {}, {}
        for name in task:
            self.labels['name'] = label_df[f"{name}_label"].to_list()
            self.number_per_class[name] = {}
            for value in set(self.labels['name']):
                if value != -1:
                    self.number_per_class[name][str(value)] = len(label_df[f"{name}_label"==value].to_list())
        
        num_per_class_age = torch.FloatTensor([
                        len(label_df[label_df['age_label'] == 0]),
                        len(label_df[label_df['age_label'] == 1]),
                        len(label_df[label_df['age_label'] == 2]),
                        len(label_df[label_df['age_label'] == 3]),
                        len(label_df[label_df['age_label'] == 4]),
                        len(label_df[label_df['age_label'] == 5])
                                                    ])
        print("Age Class count: ", self.num_per_class_age)

        num_per_class_emotion = torch.FloatTensor([
                        len(label_df[label_df['emotion_label'] == 0]),
                        len(label_df[label_df['emotion_label'] == 1]),
                        len(label_df[label_df['emotion_label'] == 2]),
                        len(label_df[label_df['emotion_label'] == 3]),
                        len(label_df[label_df['emotion_label'] == 4])
                                                        ])
        
        print("Emotion Class count: ", self.num_per_class_emotion)

        self.class_statistics = {
            "age": num_per_class_age,
            "emotion": num_per_class_emotion
        }
        
    def __len__(self):
        return len(self.item_id_list)
    
    def __getitem__(self, index):
        image_id = self.item_id_list[index]
        face_id = self.face_id_list[index]
        try:
            full_image_path = os.path.join(self.img_root, f'{image_id}.jpg')
            full_image = Image.open(full_image_path)

            face_coordinate = self.coordinate_per_image[image_id][face_id]
            face_image = full_image.crop(face_coordinate)

            # Context Image
            context_image = copy.deepcopy(full_image)
            draw = ImageDraw.Draw(context_image)
            for face_coord in self.coordinate_per_image[image_id]:
                draw.rectangle(face_coord, fill=(0, 0, 0))

            face_image = self.transform(face_image)
            context_image = self.transform(context_image)
            full_image = self.transform(full_image)

        except Exception as exception:
            print('Exception:', exception)
            return self.__getitem__(index + 1)
        
        for key in self.labels.keys():
            if key == 'age':
                age_label = self.labels[key][index]
            elif key == 'emotion':
                emotion_label = self.labels[key][index]

        return image_id, face_id, face_image, context_image, full_image, age_label, emotion_label