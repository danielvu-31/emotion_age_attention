
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models._utils as _utils
import torchvision.models as models

from .retinaface_modules import FPN, SSH, MobileNetV1


class ClassHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3, num_classes=2):
        super(ClassHead, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,
                                 self.num_anchors*self.num_classes,
                                 kernel_size=(1, 1),
                                 stride=1,
                                 padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, self.num_classes)


class ConfHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(ConfHead, self).__init__()

        self.num_classes = 2  # 0: Background, 1: Object
        self.conv1x1 = nn.Conv2d(inchannels,
                                 num_anchors*self.num_classes,
                                 kernel_size=(1, 1),
                                 stride=1,
                                 padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, self.num_classes)


class BboxHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead, self).__init__()

        self.conv1x1 = nn.Conv2d(inchannels,
                                 num_anchors*4,
                                 kernel_size=(1, 1),
                                 stride=1,
                                 padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead, self).__init__()

        self.conv1x1 = nn.Conv2d(inchannels,
                                 num_anchors*10,
                                 kernel_size=(1, 1),
                                 stride=1,
                                 padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 10)


class RetinaFace(nn.Module):
    def __init__(self, cfg=None, phase='train'):
        '''
        :param cfg:  Network related settings.
        :param phase: train or test.
        '''
        super(RetinaFace, self).__init__()

        self.phase = phase
        self.num_anchors = cfg['num_anchors']
        backbone = None
        if cfg['backbone'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if cfg['pretrain']:
                checkpoint = torch.load('./weights/mobilenetV1X0.25_pretrain.tar',
                                        map_location=torch.device('cpu'))
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                # load params
                backbone.load_state_dict(new_state_dict)
        elif cfg['backbone'] == 'Resnet50':
            backbone = models.resnet50(pretrained=cfg['pretrain'])
        elif cfg['backbone'] == 'Resnet18':
            backbone = models.resnet18(pretrained=cfg['pretrain'])

        if cfg['freeze_backbone']:
            print('Freezing backbone...')
            for param in backbone.parameters():
                param.requires_grad = False

        self.body = _utils.IntermediateLayerGetter(
            backbone, cfg['return_layers'])
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [in_channels_stage2 * 2,
                            in_channels_stage2 * 4,
                            in_channels_stage2 * 8]
        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.BboxHead = self._make_bbox_head(fpn_num=3,
                                             inchannels=cfg['out_channel'],
                                             num_anchors=self.num_anchors)
        self.LandmarkHead = self._make_landmark_head(fpn_num=3,
                                                     inchannels=cfg['out_channel'],
                                                     num_anchors=self.num_anchors)
        self.ClassHead = self._make_class_head(fpn_num=3,
                                               inchannels=cfg['out_channel'],
                                               num_anchors=self.num_anchors,
                                               num_classes=cfg['num_classes'])

    def _make_class_head(self, fpn_num=3, inchannels=64, num_anchors=2, num_classes=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, num_anchors, num_classes))
        return classhead

    def _make_bbox_head(self, fpn_num=3, inchannels=64, num_anchors=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, num_anchors))
        return bboxhead

    def _make_landmark_head(self, fpn_num=3, inchannels=64, num_anchors=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, num_anchors))
        return landmarkhead

    def forward(self, inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature)
                                      for i, feature in enumerate(features)],
                                     dim=1)
        classifications = torch.cat([self.ClassHead[i](feature)
                                     for i, feature in enumerate(features)],
                                    dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature)
                                     for i, feature in enumerate(features)],
                                    dim=1)

        if self.phase == 'train':
            output = (bbox_regressions,
                      classifications,
                      ldm_regressions)
        else:
            output = (bbox_regressions,
                      F.softmax(classifications, dim=-1),
                      ldm_regressions)
        return output
