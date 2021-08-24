import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import conv_bn_no_relu, conv_bn


class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()

        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel,
                                       out_channel//2,
                                       stride=1)

        self.conv5X5_1 = conv_bn(in_channel,
                                 out_channel//4,
                                 stride=1,
                                 leaky=leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4,
                                         out_channel//4,
                                         stride=1)

        self.conv7X7_2 = conv_bn(out_channel//4,
                                 out_channel//4,
                                 stride=1,
                                 leaky=leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4,
                                         out_channel//4,
                                         stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out
