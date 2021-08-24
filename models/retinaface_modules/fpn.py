import torch.nn as nn
import torch.nn.functional as F

from .blocks import conv_bn1X1, conv_bn


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()

        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0],
                                  out_channels,
                                  stride=1,
                                  leaky=leaky)
        self.output2 = conv_bn1X1(in_channels_list[1],
                                  out_channels,
                                  stride=1,
                                  leaky=leaky)
        self.output3 = conv_bn1X1(in_channels_list[2],
                                  out_channels,
                                  stride=1,
                                  leaky=leaky)

        self.merge1 = conv_bn(out_channels,
                              out_channels,
                              leaky=leaky)
        self.merge2 = conv_bn(out_channels,
                              out_channels,
                              leaky=leaky)

    def forward(self, input):
        input = list(input.values())

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        up3 = F.interpolate(output3,
                            size=[output2.size(2),
                                  output2.size(3)],
                            mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2,
                            size=[output1.size(2),
                                  output1.size(3)],
                            mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out