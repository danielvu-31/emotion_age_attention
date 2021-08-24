import torch
import torch.nn as nn
import torch.nn.functional as F

from resnest.torch import resnest50, resnest101

class ContextModule(nn.Module):
    def __init__(self,
                imagenet_pretrained,
                model_name,
                tasks,
                num_class_age,
                num_class_emotion,
                intermediate_feature,
                dropout_rate,
                num_train_blocks=1,
                num_ignore_blocks=0):
        super().__init__()
        if model_name == "resnest50":
            base = resnest50(imagenet_pretrained)
        else:
            base = resnest101(imagenet_pretrained)
        base_children = list(base.children())
        backbone = torch.nn.Sequential(
            *list(base_children)[:-2-num_ignore_blocks],
            base_children[-2]
        )
        self.num_ignore_blocks = num_ignore_blocks
        self._set_train_blocks(num_train_blocks)
        self.output_dim = backbone[-2][-1].conv3.out_channels

        # Classifiers
        self.classifier_age, self.classifier_emo = nn.Identity(), nn.Identity()
        if "age" in model_name:
            self.classifier_age = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features=self.output_dim,
                          out_features=intermediate_feature),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features=intermediate_feature,
                          out_features=num_class_age)
            )
        if "emotion" in model_name:
            self.classifier_emo = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features=self.output_dim,
                          out_features=intermediate_feature),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features=intermediate_feature,
                          out_features=num_class_emotion)
            )
        
        self.shared_conv = nn.Sequential(backbone[0:4])
        




    
    def _set_train_blocks(self, num_train_blocks):
        for p in self.backbone.parameters():
            p.requires_grad = False
        if num_train_blocks > 0:
            unfreeze_layers = list(self.backbone.children())[-(num_train_blocks+1):]
            for last_layer in unfreeze_layers:
                for p in last_layer.parameters():
                    p.requires_grad = True
    
        
        
