import torch.nn as nn


class SSDHead(nn.Module):

    def __init__(self, fmap_chan, priors_cnt, classes_cnt, kernel, stride, pad):
        super(SSDHead, self).__init__()
        classification_chan = classes_cnt * priors_cnt
        regression_chan = 4 * priors_cnt
        self.conv_clf = nn.Conv2d(fmap_chan, classification_chan, kernel_size=(kernel, kernel),
                                  stride=(stride, stride), padding=(pad, pad), bias=False)
        self.conv_rgr = nn.Conv2d(fmap_chan, regression_chan, kernel_size=(kernel, kernel),
                                  stride=(stride, stride), padding=(pad, pad), bias=False)
        nn.init.kaiming_uniform_(self.conv_clf.weight)
        nn.init.kaiming_uniform_(self.conv_rgr.weight)
        pass

    def forward(self, x):
        return self.conv_clf(x), self.conv_rgr(x)
