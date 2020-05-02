import torch.nn as nn
import torch


class ConvBNRelu(nn.Module):

    def __init__(self, in_chan, out_chan, kernel=3, stride=2, pad=1):
        super(ConvBNRelu, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=(kernel, kernel), stride=(stride, stride),
                              padding=(pad, pad), bias=False)
        self.bn = nn.BatchNorm2d(out_chan, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act = nn.ReLU6(inplace=True)
        pass

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

    def init_from_list(self, pars):
        self.conv.weight = pars[0]
        self.bn.weight = pars[1]
        self.bn.bias = pars[2]
        pass

    def enable_grad(self, en):
        self.conv.weight.requires_grad = en
        self.bn.weight.requires_grad = en
        self.bn.bias.requires_grad = en
        pass


class ConvMobilenetv1(nn.Module):

    def __init__(self, in_chan, out_chan, kernel, stride, pad):
        super(ConvMobilenetv1, self).__init__()
        # depthwise
        self.conv_dw = nn.Conv2d(in_chan, in_chan, kernel_size=(kernel, kernel), stride=(stride, stride),
                                 groups=in_chan, bias=False, padding=(pad, pad))
        self.bn_dw = nn.BatchNorm2d(in_chan, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act_dw = nn.ReLU6(inplace=True)
        # pointwise
        self.conv_pw_out = nn.Conv2d(in_chan, out_chan, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn_pw_out = nn.BatchNorm2d(out_chan, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act_pw = nn.ReLU6(inplace=True)
        # init
        torch.nn.init.kaiming_uniform_(self.conv_dw.weight)
        torch.nn.init.kaiming_uniform_(self.conv_pw_out.weight)
        return

    def forward(self, x):
        in_act = x
        x_dw = self.act_dw(self.bn_dw(self.conv_dw(x)))
        x_pw_o = self.act_pw(self.bn_pw_out(self.conv_pw_out(x_dw)))
        return x_pw_o


class Bottleneck(nn.Module):

    def __init__(self, in_chan, t_factor, out_chan, stride):
        super(Bottleneck, self).__init__()
        self.t = t_factor
        self.cin = in_chan
        self.cout = out_chan
        self.s = stride
        # input pointwise
        self.conv_pw = None
        self.bn_pw = None
        self.act_pw = None
        if t_factor != 1:
            self.conv_pw = nn.Conv2d(in_chan, in_chan * t_factor, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.bn_pw = nn.BatchNorm2d(in_chan * t_factor, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            self.act_pw = nn.ReLU6(inplace=True)
        # depthwise
        self.conv_dw = nn.Conv2d(in_chan * t_factor, in_chan * t_factor, kernel_size=(3, 3), stride=(stride, stride),
                                 groups=in_chan * t_factor, bias=False, padding=(1, 1))
        self.bn_dw = nn.BatchNorm2d(in_chan * t_factor, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act_dw = nn.ReLU6(inplace=True)
        # output pointwise
        self.conv_pw_out = nn.Conv2d(in_chan * t_factor, out_chan, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn_pw_out = nn.BatchNorm2d(out_chan, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.act_pw_out = nn.ReLU6(inplace=True)
        pass

    def forward(self, x):
        in_act = x
        # first pointwise convolution
        x_pw = self.act_pw(self.bn_pw(self.conv_pw(x))) if self.t != 1 else x
        x_dw = self.act_dw(self.bn_dw(self.conv_dw(x_pw)))
        # x_pw_o = self.act_pw_out(self.bn_pw_out(self.conv_pw_out(x_dw)))
        x_pw_o = self.bn_pw_out(self.conv_pw_out(x_dw))
        # residual connection
        out_act = torch.add(x_pw_o, in_act) if (self.cin == self.cout) and (self.s == 1) else x_pw_o
        return out_act

    def init_from_list(self, pars):
        if self.t != 1:
            self.conv_pw.weight = pars[0]
            self.bn_pw.weight = pars[1]
            self.bn_pw.bias = pars[2]
            self.conv_dw.weight = pars[3]
            self.bn_dw.weight = pars[4]
            self.bn_dw.bias = pars[5]
            self.conv_pw_out.weight = pars[6]
            self.bn_pw_out.weight = pars[7]
            self.bn_pw_out.bias = pars[8]
        else:
            self.conv_dw.weight = pars[0]
            self.bn_dw.weight = pars[1]
            self.bn_dw.bias = pars[2]
            self.conv_pw_out.weight = pars[3]
            self.bn_pw_out.weight = pars[4]
            self.bn_pw_out.bias = pars[5]
        pass

    def enable_grad(self, en):
        if self.t != 1:
            self.conv_pw.weight.requires_grad = en
            self.bn_pw.weight.requires_grad = en
            self.bn_pw.bias.requires_grad = en
        self.conv_dw.weight.requires_grad = en
        self.bn_dw.weight.requires_grad = en
        self.bn_dw.bias.requires_grad = en
        self.conv_pw_out.weight.requires_grad = en
        self.bn_pw_out.weight.requires_grad = en
        self.bn_pw_out.bias.requires_grad = en
        pass
