import torch
import torch.nn as nn

from resnest.torch import resnest50, resnest101
from resnest.torch.models.resnet import Bottleneck 
from resnest.torch.models.splat import SplAtConv2d


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)


class ContextModule(nn.Module):
    def __init__(self,
                imagenet_pretrained,
                model_name,
                tasks,
                num_class_age,
                num_class_emotion,
                intermediate_feature,
                dropout_rate,
                num_train_blocks=1):
        super().__init__()
        ch = [256, 512, 1024, 2048]
        if model_name == "resnest50":
            base = resnest50(imagenet_pretrained)
        else:
            base = resnest101(imagenet_pretrained)
        
        self.output_dim = base.layer4[-1].conv3.out_channels

        base_children = list(base.children())
        self.backbone = torch.nn.Sequential(
            *list(base_children)[:-1]
        )

        self._set_train_blocks(num_train_blocks)
        self.tasks = tasks

        # Classifiers
        self.classifier_age, self.classifier_emo = nn.Identity(), nn.Identity()
        if "age" in tasks:
            self.classifier_age = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features=self.output_dim,
                          out_features=intermediate_feature),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features=intermediate_feature,
                          out_features=num_class_age)
            )
        if "emotion" in tasks:
            self.classifier_emotion = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features=self.output_dim,
                          out_features=intermediate_feature),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features=intermediate_feature,
                          out_features=num_class_emotion)
            )
        
        self.shared_conv = nn.Sequential(self.backbone[0:4])

        self.shared_layer1_b = self.backbone[4][:-1]
        self.shared_layer1_t = self.backbone[4][-1]

        self.shared_layer2_b = self.backbone[5][:-1]
        self.shared_layer2_t = self.backbone[5][-1]

        self.shared_layer3_b = self.backbone[6][:-1]
        self.shared_layer3_t = self.backbone[6][-1]

        self.shared_layer4_b = self.backbone[7][:-1]
        self.shared_layer4_t = self.backbone[7][-1]

        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pooling = GlobalAvgPool2d()

        # Define task specific attention modules (Bottleneck design)
        self.encoder_att_1 = nn.ModuleList([self.att_layer(ch[0]) for _ in self.tasks])
        self.encoder_att_2 = nn.ModuleList([self.att_layer(ch[1], 2) for _ in self.tasks])
        self.encoder_att_3 = nn.ModuleList([self.att_layer(ch[2], 2) for _ in self.tasks])
        self.encoder_att_4 = nn.ModuleList([self.att_layer(ch[3], 2) for _ in self.tasks])

        # Sharing modules between 2 tasks (except last bottleneck)
        self.encoder_block_att_1 = self.conv_layer(int(ch[1]/2), int(ch[1]/4), 2, True, True)
        self.encoder_block_att_2 = self.conv_layer(int(ch[2]/2), int(ch[2]/4), 2, True, True)
        self.encoder_block_att_3 = self.conv_layer(int(ch[3]/2), int(ch[3]/4), 2, True, True)

    # Last layer
    def att_layer(self, in_channel, scale=1): # Add scale due to concatenation
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel*scale, out_channels=int(in_channel/4), kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(int(in_channel/4)),
            SplAtConv2d(in_channels=int(in_channel/4),
                        channels=int(in_channel/4),
                        kernel_size=3,
                        padding=1,
                        bias=False,
                        norm_layer=nn.BatchNorm2d),
            nn.Conv2d(in_channels=int(in_channel/4), out_channels=in_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.Sigmoid()
            )
    
    def conv_layer(self, in_channel, inter_channel, scale, avd, is_first):
        downsample = nn.Sequential(
            nn.AvgPool2d(kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channel, in_channel*scale, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channel*scale)
        )
        return Bottleneck(inplanes=in_channel,
                          planes=inter_channel,
                          norm_layer=nn.BatchNorm2d,
                          radix=2,
                          avd=avd,
                          is_first=is_first,
                          downsample=downsample)

    def _set_train_blocks(self, num_train_blocks):
        for p in self.backbone.parameters():
            p.requires_grad = False
        if num_train_blocks > 0:
            unfreeze_layers = list(self.backbone.children())[-(num_train_blocks+1):]
            for last_layer in unfreeze_layers:
                for p in last_layer.parameters():
                    p.requires_grad = True
    
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
        a_1 = [self.down_sampling(self.encoder_block_att_1(a_1_i)) for a_1_i in a_1]

        # Attention block 2
        a_2_mask = [att_i(torch.cat((u_2_b, a_1_i), dim=1)) for a_1_i, att_i in zip(a_1, self.encoder_att_2)] # Concat input and attention adjusted
        a_2 = [a_2_mask_i * u_2_t for a_2_mask_i in a_2_mask]
        a_2 = [self.down_sampling(self.encoder_block_att_2(a_2_i)) for a_2_i in a_2]

        # Attention block 3 -> Apply attention over last residual block
        a_3_mask = [att_i(torch.cat((u_3_b, a_2_i), dim=1)) for a_2_i, att_i in zip(a_2, self.encoder_att_3)]
        a_3 = [a_3_mask_i * u_3_t for a_3_mask_i in a_3_mask]
        a_3 = [self.down_sampling(self.encoder_block_att_3(a_3_i)) for a_3_i in a_3]
        
        # Attention block 4 -> Apply attention over last residual block (without final encoder)
        a_4_mask = [att_i(torch.cat((u_4_b, a_3_i), dim=1)) for a_3_i, att_i in zip(a_3, self.encoder_att_4)]
        a_4 = [self.pooling(a_4_mask_i * u_4_t) for a_4_mask_i in a_4_mask]

        age_output = self.classifier_age(a_4)
        emotion_output = self.classifier_emotion(a_4)
        
        return age_output, emotion_output
    
        
if __name__ == '__main__':
    ctx = ContextModule(imagenet_pretrained=False,
                        model_name="resnest50",
                        tasks=["age","emotion"],
                        num_class_age=6,
                        num_class_emotion=5,
                        intermediate_feature=256, dropout_rate=0.5,
                        num_train_blocks=0)
    ctx.eval()

