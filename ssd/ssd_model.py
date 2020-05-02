import torch.nn as nn
from ssd.mobilenetv2 import Mobilenetv2Backbone
from ssd.ssd_head import SSDHead
import numpy as np
import torch
import unittest
from common_utils.tensor_transform import generate_random_tensor


class SSD(nn.Module):

    def __init__(self, priors_cnt=6, classes_cnt=21, pretrained=True, requires_grad=False):
        super(SSD, self).__init__()
        self.priors_cnt = priors_cnt
        self.classes_cnt = classes_cnt
        self.backbone = Mobilenetv2Backbone(pretrained=pretrained, requires_grad=requires_grad)
        self.head_0 = SSDHead(32, priors_cnt, classes_cnt, kernel=3, stride=1, pad=1)
        self.head_1 = SSDHead(96, priors_cnt, classes_cnt, kernel=3, stride=1, pad=1)
        self.head_2 = SSDHead(320, priors_cnt, classes_cnt, kernel=3, stride=1, pad=1)
        self.head_3 = SSDHead(480, priors_cnt, classes_cnt, kernel=3, stride=1, pad=1)
        self.head_4 = SSDHead(640, priors_cnt, classes_cnt, kernel=3, stride=1, pad=1)
        self.head_5 = SSDHead(640, priors_cnt, classes_cnt, kernel=1, stride=1, pad=0)
        return

    def forward(self, x):
        feature_maps = self.backbone.forward(x)
        clf0, rgr0 = self.head_0(feature_maps[0])
        clf1, rgr1 = self.head_1(feature_maps[1])
        clf2, rgr2 = self.head_2(feature_maps[2])
        clf3, rgr3 = self.head_3(feature_maps[3])
        clf4, rgr4 = self.head_4(feature_maps[4])
        clf5, rgr5 = self.head_5(feature_maps[5])
        return clf0, rgr0, clf1, rgr1, clf2, rgr2, clf3, rgr3, \
               clf4, rgr4, clf5, rgr5

    def __str__(self):
        return "SSD_Mobilenetv2_6fm" + str(self.priors_cnt) + "p" + \
               str(self.classes_cnt) + "c"


class TestSSDBasics(unittest.TestCase):

    def test_forward_pass(self):
        batch_sz = 4
        model = SSD()
        test_in = torch.from_numpy(np.arange(300*300*3*batch_sz).reshape(batch_sz, 3, 300, 300).astype(np.float32))
        net_out = model.forward(test_in)
        self.assertEqual(len(net_out), 12)
        for out in net_out:
            self.assertEqual(out.shape[0], batch_sz)
        return

    def test_back_prop(self):
        batch_sz = 8
        priors = 6
        classes = 21
        model = SSD(priors_cnt=priors, classes_cnt=classes)
        test_in = torch.from_numpy(np.arange(300*300*3*batch_sz).reshape(batch_sz, 3, 300, 300).astype(np.float32))
        clf0_tgt = generate_random_tensor(batch_sz, priors * classes, 38, 38)
        reg0_tgt = generate_random_tensor(batch_sz, priors * 4, 38, 38)
        clf1_tgt = generate_random_tensor(batch_sz, priors * classes, 19, 19)
        reg1_tgt = generate_random_tensor(batch_sz, priors * 4, 19, 19)
        clf2_tgt = generate_random_tensor(batch_sz, priors * classes, 10, 10)
        reg2_tgt = generate_random_tensor(batch_sz, priors * 4, 10, 10)
        clf3_tgt = generate_random_tensor(batch_sz, priors * classes, 5, 5)
        reg3_tgt = generate_random_tensor(batch_sz, priors * 4, 5, 5)
        clf4_tgt = generate_random_tensor(batch_sz, priors * classes, 3, 3)
        reg4_tgt = generate_random_tensor(batch_sz, priors * 4, 3, 3)
        clf5_tgt = generate_random_tensor(batch_sz, priors * classes, 1, 1)
        reg5_tgt = generate_random_tensor(batch_sz, priors * 4, 1, 1)

        model.train()
        net_out = model.forward(test_in)
        target_out = (clf0_tgt, reg0_tgt, clf1_tgt, reg1_tgt, clf2_tgt, reg2_tgt, clf3_tgt, reg3_tgt,
                      clf4_tgt, reg4_tgt, clf5_tgt, reg5_tgt)
        l0 = torch.nn.functional.mse_loss(net_out[0], target_out[0])
        l1 = torch.nn.functional.mse_loss(net_out[1], target_out[1])
        l2 = torch.nn.functional.mse_loss(net_out[2], target_out[2])
        l3 = torch.nn.functional.mse_loss(net_out[3], target_out[3])
        l4 = torch.nn.functional.mse_loss(net_out[4], target_out[4])
        l5 = torch.nn.functional.mse_loss(net_out[5], target_out[5])
        l6 = torch.nn.functional.mse_loss(net_out[6], target_out[6])
        l7 = torch.nn.functional.mse_loss(net_out[7], target_out[7])
        l8 = torch.nn.functional.mse_loss(net_out[8], target_out[8])
        l9 = torch.nn.functional.mse_loss(net_out[9], target_out[9])
        l10 = torch.nn.functional.mse_loss(net_out[10], target_out[10])
        l11 = torch.nn.functional.mse_loss(net_out[11], target_out[11])

        total = l0+l1+l2+l3+l4+l5+l6+l7+l8+l9+l10+l11
        total.backward()
        return


if __name__ == "__main__":
    unittest.main()
