import torch.nn as nn
from torchvision import models
from ssd.layers import *


class MobilenetV2FeatureExtractor(nn.Module):

    def __init__(self, pretrained=True, requires_grad=False):
        super(MobilenetV2FeatureExtractor, self).__init__()

        self.pretrained = pretrained
        self.requires_grad = requires_grad

        # architecture

        # input tensor - 3x224x224
        self.conv0 = ConvBNRelu(in_chan=3, out_chan=32, kernel=3, stride=2, pad=1)          # 32x112x112
        self.bottleneck1 = Bottleneck(in_chan=32, t_factor=1, out_chan=16, stride=1)        # 16x112x112
        self.bottleneck2 = Bottleneck(in_chan=16, t_factor=6, out_chan=24, stride=2)        # 24x56x56
        self.bottleneck3 = Bottleneck(in_chan=24, t_factor=6, out_chan=24, stride=1)        # 24x56x56
        self.bottleneck4 = Bottleneck(in_chan=24, t_factor=6, out_chan=32, stride=2)        # 32x28x28
        self.bottleneck5 = Bottleneck(in_chan=32, t_factor=6, out_chan=32, stride=1)        # 32x28x28
        self.bottleneck6 = Bottleneck(in_chan=32, t_factor=6, out_chan=32, stride=1)        # 32x28x28
        self.bottleneck7 = Bottleneck(in_chan=32, t_factor=6, out_chan=64, stride=2)        # 64x14x14
        self.bottleneck8 = Bottleneck(in_chan=64, t_factor=6, out_chan=64, stride=1)        # 64x14x14
        self.bottleneck9 = Bottleneck(in_chan=64, t_factor=6, out_chan=64, stride=1)        # 64x14x14
        self.bottleneck10 = Bottleneck(in_chan=64, t_factor=6, out_chan=64, stride=1)       # 64x14x14
        self.bottleneck11 = Bottleneck(in_chan=64, t_factor=6, out_chan=96, stride=1)       # 96x14x14
        self.bottleneck12 = Bottleneck(in_chan=96, t_factor=6, out_chan=96, stride=1)       # 96x14x14
        self.bottleneck13 = Bottleneck(in_chan=96, t_factor=6, out_chan=96, stride=1)       # 96x14x14
        self.bottleneck14 = Bottleneck(in_chan=96, t_factor=6, out_chan=160, stride=2)      # 160x7x7
        self.bottleneck15 = Bottleneck(in_chan=160, t_factor=6, out_chan=160, stride=1)     # 160x7x7
        self.bottleneck16 = Bottleneck(in_chan=160, t_factor=6, out_chan=160, stride=1)     # 160x7x7
        self.bottleneck17 = Bottleneck(in_chan=160, t_factor=6, out_chan=320, stride=1)     # 320x7x7
        self.conv18 = ConvBNRelu(in_chan=320, out_chan=1280, kernel=1, stride=1, pad=0)            # 1280x7x7

        self.pars_count_per_layer = [3, 6]
        self.pars_count_per_layer.extend([9] * 16)
        self.pars_count_per_layer.extend([3])

        # initialization

        if self.pretrained is not None:
            if self.pretrained:
                mobilenetv2 = models.mobilenet_v2(pretrained=True)
                w_pretrained = list(mobilenetv2.parameters())

                start = 0
                stop = start + self.pars_count_per_layer[0]
                sublist = w_pretrained[start: stop]
                self.conv0.init_from_list(sublist)
                self.conv0.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[1]
                sublist = w_pretrained[start: stop]
                self.bottleneck1.init_from_list(sublist)
                self.bottleneck1.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[2]
                sublist = w_pretrained[start: stop]
                self.bottleneck2.init_from_list(sublist)
                self.bottleneck2.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[3]
                sublist = w_pretrained[start: stop]
                self.bottleneck3.init_from_list(sublist)
                self.bottleneck3.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[4]
                sublist = w_pretrained[start: stop]
                self.bottleneck4.init_from_list(sublist)
                self.bottleneck4.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[5]
                sublist = w_pretrained[start: stop]
                self.bottleneck5.init_from_list(sublist)
                self.bottleneck5.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[6]
                sublist = w_pretrained[start: stop]
                self.bottleneck6.init_from_list(sublist)
                self.bottleneck6.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[7]
                sublist = w_pretrained[start: stop]
                self.bottleneck7.init_from_list(sublist)
                self.bottleneck7.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[8]
                sublist = w_pretrained[start: stop]
                self.bottleneck8.init_from_list(sublist)
                self.bottleneck8.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[9]
                sublist = w_pretrained[start: stop]
                self.bottleneck9.init_from_list(sublist)
                self.bottleneck9.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[10]
                sublist = w_pretrained[start: stop]
                self.bottleneck10.init_from_list(sublist)
                self.bottleneck10.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[11]
                sublist = w_pretrained[start: stop]
                self.bottleneck11.init_from_list(sublist)
                self.bottleneck11.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[12]
                sublist = w_pretrained[start: stop]
                self.bottleneck12.init_from_list(sublist)
                self.bottleneck12.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[13]
                sublist = w_pretrained[start: stop]
                self.bottleneck13.init_from_list(sublist)
                self.bottleneck13.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[14]
                sublist = w_pretrained[start: stop]
                self.bottleneck14.init_from_list(sublist)
                self.bottleneck14.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[15]
                sublist = w_pretrained[start: stop]
                self.bottleneck15.init_from_list(sublist)
                self.bottleneck15.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[16]
                sublist = w_pretrained[start: stop]
                self.bottleneck16.init_from_list(sublist)
                self.bottleneck16.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[17]
                sublist = w_pretrained[start: stop]
                self.bottleneck17.init_from_list(sublist)
                self.bottleneck17.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[18]
                sublist = w_pretrained[start: stop]
                self.conv18.init_from_list(sublist)
                self.conv18.enable_grad(requires_grad)
        pass

    def forward(self, x):
        x = self.conv0(x)
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)
        x = self.bottleneck5(x)
        x = self.bottleneck6(x)
        x = self.bottleneck7(x)
        x = self.bottleneck8(x)
        x = self.bottleneck9(x)
        x = self.bottleneck10(x)
        x = self.bottleneck11(x)
        x = self.bottleneck12(x)
        x = self.bottleneck13(x)
        x = self.bottleneck14(x)
        x = self.bottleneck15(x)
        x = self.bottleneck16(x)
        x = self.bottleneck17(x)
        x = self.conv18(x)
        return x


class Mobilenetv2Backbone(nn.Module):

    def __init__(self, pretrained=True, requires_grad=False):
        super(Mobilenetv2Backbone, self).__init__()

        self.pretrained = pretrained
        self.requires_grad = requires_grad

        # architecture

        # input tensor - 3x300x300
        self.conv0 = ConvBNRelu(in_chan=3, out_chan=32, kernel=3, stride=2, pad=1)          # 32x150x150
        self.bottleneck1 = Bottleneck(in_chan=32, t_factor=1, out_chan=16, stride=1)        # 16x150x150
        self.bottleneck2 = Bottleneck(in_chan=16, t_factor=6, out_chan=24, stride=2)        # 24x75x75
        self.bottleneck3 = Bottleneck(in_chan=24, t_factor=6, out_chan=24, stride=1)        # 24x75x75
        self.bottleneck4 = Bottleneck(in_chan=24, t_factor=6, out_chan=32, stride=2)        # 32x38x38
        self.bottleneck5 = Bottleneck(in_chan=32, t_factor=6, out_chan=32, stride=1)        # 32x38x38
        self.bottleneck6 = Bottleneck(in_chan=32, t_factor=6, out_chan=32, stride=1)        # 32x38x38  --> 0
        self.bottleneck7 = Bottleneck(in_chan=32, t_factor=6, out_chan=64, stride=2)        # 64x19x19
        self.bottleneck8 = Bottleneck(in_chan=64, t_factor=6, out_chan=64, stride=1)        # 64x19x19
        self.bottleneck9 = Bottleneck(in_chan=64, t_factor=6, out_chan=64, stride=1)        # 64x19x19
        self.bottleneck10 = Bottleneck(in_chan=64, t_factor=6, out_chan=64, stride=1)       # 64x19x19
        self.bottleneck11 = Bottleneck(in_chan=64, t_factor=6, out_chan=96, stride=1)       # 96x19x19
        self.bottleneck12 = Bottleneck(in_chan=96, t_factor=6, out_chan=96, stride=1)       # 96x19x19
        self.bottleneck13 = Bottleneck(in_chan=96, t_factor=6, out_chan=96, stride=1)       # 96x19x19  --> 1
        self.bottleneck14 = Bottleneck(in_chan=96, t_factor=6, out_chan=160, stride=2)      # 160x10x10
        self.bottleneck15 = Bottleneck(in_chan=160, t_factor=6, out_chan=160, stride=1)     # 160x10x10
        self.bottleneck16 = Bottleneck(in_chan=160, t_factor=6, out_chan=160, stride=1)     # 160x10x10
        self.bottleneck17 = Bottleneck(in_chan=160, t_factor=6, out_chan=320, stride=1)     # 320x10x10  --> 2

        # additional layers for 3 last scales
        self.dwconv18 = ConvMobilenetv1(in_chan=320, out_chan=480, kernel=3, stride=2, pad=1)  # 480x5x5 --> 3
        self.dwconv19 = ConvMobilenetv1(in_chan=480, out_chan=640, kernel=3, stride=1, pad=0)  # 640x3x3 --> 4
        self.dwconv20 = ConvMobilenetv1(in_chan=640, out_chan=640, kernel=3, stride=1, pad=0)  # 640x1x1 --> 5
        # self.avgpool20 = nn.AvgPool2d(kernel_size=3)                                           # 640x1x1 --> 5

        self.pars_count_per_layer = [3, 6]
        self.pars_count_per_layer.extend([9] * 17)

        # initialization

        if self.pretrained is not None:
            if self.pretrained:
                mobilenetv2 = models.mobilenet_v2(pretrained=True)
                w_pretrained = list(mobilenetv2.parameters())

                start = 0
                stop = start + self.pars_count_per_layer[0]
                sublist = w_pretrained[start: stop]
                self.conv0.init_from_list(sublist)
                self.conv0.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[1]
                sublist = w_pretrained[start: stop]
                self.bottleneck1.init_from_list(sublist)
                self.bottleneck1.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[2]
                sublist = w_pretrained[start: stop]
                self.bottleneck2.init_from_list(sublist)
                self.bottleneck2.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[3]
                sublist = w_pretrained[start: stop]
                self.bottleneck3.init_from_list(sublist)
                self.bottleneck3.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[4]
                sublist = w_pretrained[start: stop]
                self.bottleneck4.init_from_list(sublist)
                self.bottleneck4.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[5]
                sublist = w_pretrained[start: stop]
                self.bottleneck5.init_from_list(sublist)
                self.bottleneck5.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[6]
                sublist = w_pretrained[start: stop]
                self.bottleneck6.init_from_list(sublist)
                self.bottleneck6.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[7]
                sublist = w_pretrained[start: stop]
                self.bottleneck7.init_from_list(sublist)
                self.bottleneck7.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[8]
                sublist = w_pretrained[start: stop]
                self.bottleneck8.init_from_list(sublist)
                self.bottleneck8.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[9]
                sublist = w_pretrained[start: stop]
                self.bottleneck9.init_from_list(sublist)
                self.bottleneck9.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[10]
                sublist = w_pretrained[start: stop]
                self.bottleneck10.init_from_list(sublist)
                self.bottleneck10.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[11]
                sublist = w_pretrained[start: stop]
                self.bottleneck11.init_from_list(sublist)
                self.bottleneck11.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[12]
                sublist = w_pretrained[start: stop]
                self.bottleneck12.init_from_list(sublist)
                self.bottleneck12.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[13]
                sublist = w_pretrained[start: stop]
                self.bottleneck13.init_from_list(sublist)
                self.bottleneck13.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[14]
                sublist = w_pretrained[start: stop]
                self.bottleneck14.init_from_list(sublist)
                self.bottleneck14.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[15]
                sublist = w_pretrained[start: stop]
                self.bottleneck15.init_from_list(sublist)
                self.bottleneck15.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[16]
                sublist = w_pretrained[start: stop]
                self.bottleneck16.init_from_list(sublist)
                self.bottleneck16.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[17]
                sublist = w_pretrained[start: stop]
                self.bottleneck17.init_from_list(sublist)
                self.bottleneck17.enable_grad(requires_grad)
        pass

    def forward(self, x):
        x = self.conv0(x)
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)
        x = self.bottleneck5(x)
        out_0 = self.bottleneck6(x)     # 32x28x28  --> 0
        x = self.bottleneck7(out_0)
        x = self.bottleneck8(x)
        x = self.bottleneck9(x)
        x = self.bottleneck10(x)
        x = self.bottleneck11(x)
        x = self.bottleneck12(x)
        out_1 = self.bottleneck13(x)    # 96x14x14  --> 1
        x = self.bottleneck14(out_1)
        x = self.bottleneck15(x)
        x = self.bottleneck16(x)
        out_2 = self.bottleneck17(x)    # 320x7x7  --> 2
        out_3 = self.dwconv18(out_2)    # 480x5x5 --> 3
        out_4 = self.dwconv19(out_3)    # 640x3x3 --> 4
        # out_5 = self.avgpool20(out_4)   # 640x1x1 --> 5
        out_5 = self.dwconv20(out_4)
        return out_0, out_1, out_2, out_3, out_4, out_5


