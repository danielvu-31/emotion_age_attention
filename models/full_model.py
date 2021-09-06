import torch
import torch.nn as nn
import copy

from .modules import FaceModel, ContextModule
from base import BaseModel


class GatingModule(nn.Module):
    def __init__(self, context_model):
        super().__init__()
        self.backbone = context_model
        self.gating = nn.Sigmoid()

    def forward(self, x):
        out_age, out_emotion = self.backbone(x)
        return self.gating(out_age), self.gating(out_emotion)


class ContextAwareAttention(BaseModel):
    def __init__(self,
                 tasks,
                 intermediate_feature,
                 num_age_classes,
                 num_emotion_classes,
                 face_model_name,
                 face_dropout,
                 face_backbone_path,
                 context_pretrained, # True or False for ImageNet ResNest pretrained
                 context_model_name,
                 context_dropout,
                 forward_phase,
                 mode):

        super().__init__()
        self.forward_phase = forward_phase

        self.face_module = FaceModel(tasks,
                                     face_model_name,
                                     True,
                                     intermediate_feature,
                                     face_dropout,
                                     num_emotion_classes,
                                     num_age_classes,
                                     mode,
                                     face_backbone_path)

        self.context_module = ContextModule(context_pretrained,
                                            context_model_name,
                                            tasks,
                                            num_age_classes,
                                            num_emotion_classes,
                                            intermediate_feature,
                                            context_dropout,
                                            num_train_blocks=1)

        context_for_gating = copy.deepcopy(self.context_module)
        self.gating_module = GatingModule(context_for_gating)

        self.num_emotion_classes = num_emotion_classes
        self.num_age_classes = num_age_classes

    def forward(self, face_image, context_image, full_image):
        if self.forward_phase == 'face':
            face_age, face_emotion = self.face_module(face_image)
            return face_age, face_emotion

        elif self.forward_phase == 'context':
            context_age, context_emotion = self.context_module(context_image)
            return context_age, context_emotion

        else:
            face_age, face_emotion = self.face_module(face_image)
            context_age, context_emotion = self.context_module(context_image)
            gating_age, gating_emotion = self.gating_module(full_image)

            ones_emotion = torch.ones(self.num_emotion_classes).to(face_emotion.device)
            ones_age = torch.ones(self.num_age_classes).to(face_age.device)

            out_age = face_age * (ones_age - gating_age) + context_age * gating_age
            out_emotion = face_emotion * (ones_emotion - gating_emotion) + context_emotion * gating_emotion

            return out_age, out_emotion
