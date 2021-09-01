import torch.nn.functional as F
import torch.nn as nn

import torch
from arcface_pytorch.iresnet import iresnet_general, IBasicBlock
from base import BaseModel


class FaceModel(BaseModel):
    def __init__(self,
                tasks,
                model_name,
                init_xavier,
                intermediate_feature,
                dropout_rate,
                num_emotion_classes,
                num_age_classes,
                phase,
                face_recognition_backbone_path=None):
        super().__init__()
        ch = [64, 128, 256, 512]

        backbone = iresnet_general(model_name)
        self.output_dim = backbone.fc.out_features
        self.tasks = tasks

        if face_recognition_backbone_path is not None:
            state_dict = torch.load(face_recognition_backbone_path, "cpu")
            backbone.load_state_dict(state_dict)
        
        self.shared_conv = nn.Sequential(backbone.conv1, backbone.bn1, backbone.prelu)
        self.shared_layer1_b = backbone.layer1[:-1]
        self.shared_layer1_t = backbone.layer1[-1]

        self.shared_layer2_b = backbone.layer2[:-1]
        self.shared_layer2_t = backbone.layer2[-1]

        self.shared_layer3_b = backbone.layer3[:-1]
        self.shared_layer3_t = backbone.layer3[-1]

        self.shared_layer4_b = backbone.layer4[:-1]
        self.shared_layer4_t = backbone.layer4[-1]

        # Define task specific attention modules using a similar bottleneck design in residual block
        # (to avoid large computations)
        self.encoder_att_1 = nn.ModuleList([self.att_layer(ch[0], 1) for _ in self.tasks])
        self.encoder_att_2 = nn.ModuleList([self.att_layer(ch[1], 2) for _ in self.tasks])
        self.encoder_att_3 = nn.ModuleList([self.att_layer(ch[2], 2) for _ in self.tasks])
        self.encoder_att_4 = nn.ModuleList([self.att_layer(ch[3], 2) for _ in self.tasks])

        # Define task shared attention encoders using residual bottleneck layers
        # We do not apply shared attention encoders at the last layer,
        # so the attended features will be directly fed into the task-specific decoders.
        self.encoder_block_att_1 = self.conv_layer(ch[0], 2)
        self.encoder_block_att_2 = self.conv_layer(ch[1], 2)
        self.encoder_block_att_3 = self.conv_layer(ch[2], 2)
        
        # self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2)

        # Classifiers
        self.classifier_age, self.classifier_emotion = nn.Identity(), nn.Identity()
        if "age" in model_name:
            self.classifier_age = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features=self.output_dim,
                          out_features=intermediate_feature),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features=intermediate_feature,
                          out_features=num_age_classes)
            )
        if "emotion" in model_name:
            self.classifier_emotion = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features=self.output_dim,
                          out_features=intermediate_feature),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features=intermediate_feature,
                          out_features=num_emotion_classes)
            )
        
        if phase == 'train':
            if init_xavier:
                self.classifier_age = self._init_weights_xavier(self.classifier_age)
                self.classifier_emotion = self._init_weights_xavier(self.classifier_emotion)
                for i in range(len(ch)):
                    setattr(self, f"encoder_att_{i+1}", self._init_weights_xavier(f"encoder_att_{i+1}"))
                    if i != 3:
                        setattr(self, f"encoder_block_att_{i+1}", self._init_weights_xavier(f"encoder_block_att_{i+1}"))

    def att_layer(self, in_channel, scale):
        return nn.Sequential(
            nn.BatchNorm2d(in_channel*scale),
            nn.Conv2d(in_channels=in_channel*scale, out_channels=in_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channel),
            nn.PReLU(num_parameters=in_channel),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channel),
            nn.Sigmoid())
    
    def conv_layer(self, in_channel, strides):
        downsample = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel*2,
                      kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channel*2)
        )
        return IBasicBlock(in_channel, in_channel*2, stride=strides, downsample=downsample)

    def _init_weights_xavier(self, model):
        '''
        The fully connected layers are both initilized with Xavier algorithm.
        In particular, we set the parameters to random values uniformly drawn from [-a, a]
        where a = sqrt(6 * (din + dout)).
        For batch normalization layers, y=1, b=0, all bias initialized to 0.
        '''
        for m in model.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        return model
    
    def forward(self, x):
        # Shared convolution
        x = self.shared_conv(x)

        # Shared ResNet block 1
        u_1_b = self.shared_layer1_b(x)
        u_1_t = self.shared_layer1_t(u_1_b)

        # Shared ResNet block 2
        u_2_b = self.shared_layer2_b(u_1_t)
        u_2_t = self.shared_layer2_t(u_2_b)

        # Shared ResNet block 3
        u_3_b = self.shared_layer3_b(u_2_t)
        u_3_t = self.shared_layer3_t(u_3_b)
        
        # Shared ResNet block 4
        u_4_b = self.shared_layer4_b(u_3_t)
        u_4_t = self.shared_layer4_t(u_4_b)

        # Attention block 1 -> Apply attention over last residual block
        a_1_mask = [att_i(u_1_b) for att_i in self.encoder_att_1]  # Generate task specific attention map
        a_1 = [a_1_mask_i * u_1_t for a_1_mask_i in a_1_mask]  # Apply task specific attention map to shared features
        a_1 = [self.encoder_block_att_1(a_1_i) for a_1_i in a_1]

        # Attention block 2
        a_2_mask = [att_i(torch.cat((u_2_b, a_1_i), dim=1)) for a_1_i, att_i in zip(a_1, self.encoder_att_2)] # Concat input and attention adjusted
        a_2 = [a_2_mask_i * u_2_t for a_2_mask_i in a_2_mask]
        a_2 = [self.encoder_block_att_2(a_2_i) for a_2_i in a_2]

        # Attention block 3 -> Apply attention over last residual block
        a_3_mask = [att_i(torch.cat((u_3_b, a_2_i), dim=1)) for a_2_i, att_i in zip(a_2, self.encoder_att_3)]
        a_3 = [a_3_mask_i * u_3_t for a_3_mask_i in a_3_mask]
        a_3 = [self.encoder_block_att_3(a_3_i) for a_3_i in a_3]
        
        # Attention block 4 -> Apply attention over last residual block (without final encoder)
        a_4_mask = [att_i(torch.cat((u_4_b, a_3_i), dim=1)) for a_3_i, att_i in zip(a_3, self.encoder_att_4)]
        a_4 = [a_4_mask_i * u_4_t for a_4_mask_i in a_4_mask]

        age_output = self.classifier_age(a_4)
        emotion_output = self.classifier_emotion(a_4)
        
        return age_output, emotion_output
    
if __name__ == '__main__':
    face = FaceModel(tasks=["age", "emotion"],
                    model_name="iresnet100",
                    init_xavier=False,
                    face_recognition_backbone_path=None,
                    intermediate_feature=256,
                    dropout_rate=0.5,
                    num_emotion_classes=5,
                    num_age_classes=6,
                    phase="train")
    face.eval()

