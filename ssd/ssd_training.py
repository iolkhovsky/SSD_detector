from torch.nn.functional import smooth_l1_loss, cross_entropy, binary_cross_entropy
from torch import log
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import unittest
import random

from common_utils.partial_sort import PartialSorter
from common_utils.logger import LogDuration
from common_utils.tensor_transform import *

""" 1. The localization loss is the mismatch between the ground truth box and the predicted boundary box.
SSD only penalizes predictions from positive matches. We want the predictions from the positive matches
to get closer to the ground truth. Negative matches can be ignored.
2. The confidence loss is the loss in making a class prediction. For every positive match prediction,
we penalize the loss according to the confidence score of the corresponding class. For negative match predictions,
we penalize the loss according to the confidence score of the class “0”: class “0” classifies no object is detected."""


def get_hard_neg_ids(predicted_clf_mismatched, hard_neg_cnt):
    predicted = predicted_clf_mismatched.cpu().detach().numpy()
    if hard_neg_cnt == 0:
        return []
    sorter = PartialSorter(hard_neg_cnt)
    for idx, distr in enumerate(predicted):
        prob = np.max(distr[:-1])
        sorter.push((prob, idx))
    probs, ids = zip(*sorter.get_data())
    return list(ids)


def mismatched_loss(hard_neg_samples):
    out = 0.
    for sample in hard_neg_samples:
        out += -1*log(sample[-1])
    return out


def head_loss_v1(target_clf, target_loc, predicted_clf, predicted_loc, npr=3., coords=4, classes=21):
    batch_size = target_loc.shape[0]
    y_size = target_loc.shape[2]
    x_size = target_loc.shape[3]

    # permute tensor b,c,y,x -> b,y,x,c
    predicted_clf = tensor_bcyx2byxc(predicted_clf)
    predicted_loc = tensor_bcyx2byxc(predicted_loc)
    target_clf = tensor_bcyx2byxc(target_clf)
    target_loc = tensor_bcyx2byxc(target_loc)

    # reshape tensors to b,y,x,anchor, coords
    predicted_loc = predicted_loc.reshape(batch_size, y_size, x_size, -1, coords)
    target_loc = target_loc.reshape(batch_size, y_size, x_size, -1, coords)
    predicted_clf = predicted_clf.reshape(batch_size, y_size, x_size, -1, classes)
    target_clf = target_clf.reshape(batch_size, y_size, x_size, -1, classes)

    predicted_clf = torch.nn.functional.softmax(predicted_clf, dim=4)

    # find all matched anchors
    obj_indices = target_clf[:, :, :, :, -1] == 0  # last probability = 0 <-> <No Onject> class
    noobj_indices = ~obj_indices

    # define all subsets by matching
    target_clf_matched = target_clf[obj_indices]
    predicted_clf_matched = predicted_clf[obj_indices]

    matched_cnt = len(target_clf_matched)
    hard_neg_cnt = int(matched_cnt * npr)

    if matched_cnt == 0:
        return torch.from_numpy(np.zeros(shape=(1,))), torch.from_numpy(np.zeros(shape=(1,))), 0

    target_loc_matched = target_loc[obj_indices]
    predicted_loc_matched = predicted_loc[obj_indices]

    predicted_clf_mismatched = predicted_clf[noobj_indices]
    worst_errors_ids = get_hard_neg_ids(predicted_clf_mismatched, hard_neg_cnt)
    hard_neg_samples = predicted_clf_mismatched[worst_errors_ids]

    regr_loss = smooth_l1_loss(predicted_loc_matched, target_loc_matched, reduction='sum')
    target_clf_matched_ids = target_clf_matched.argmax(dim=1)
    clf_loss_matched = cross_entropy(predicted_clf_matched, target_clf_matched_ids, reduction='sum')

    hard_neg_target = torch.from_numpy(np.ones(shape=(len(hard_neg_samples),), dtype=np.float32))
    hard_neg_target = hard_neg_target.to(hard_neg_samples.device)  # transfer tensor if use cuda
    clf_loss_mismatched = binary_cross_entropy(hard_neg_samples[:, -1], hard_neg_target, reduction='sum')

    clf_loss = clf_loss_matched + clf_loss_mismatched

    return clf_loss, regr_loss, matched_cnt


def head_loss_v2(target_clf, target_loc, predicted_clf, predicted_loc, npr=1., coords=4, classes=21):

    # permute tensor b,c,y,x -> b,y,x,c
    predicted_clf = tensor_bcyx2byxc(predicted_clf)
    predicted_loc = tensor_bcyx2byxc(predicted_loc)
    target_clf = tensor_bcyx2byxc(target_clf)
    target_loc = tensor_bcyx2byxc(target_loc)

    # reshape tensors to b,y,x,anchor, coords
    predicted_loc = predicted_loc.reshape(-1, coords)
    target_loc = target_loc.reshape(-1, coords)
    predicted_clf = predicted_clf.reshape(-1, classes)
    target_clf = target_clf.reshape(-1, classes)

    predicted_clf = torch.nn.functional.softmax(predicted_clf, dim=1)

    # find all matched anchors
    obj_indices = (target_clf[:, -1] == 0)  # last probability = 0 <-> <No Onject> class
    noobj_indices = ~obj_indices

    # define all subsets by matching
    target_clf_matched = target_clf[obj_indices]
    predicted_clf_matched = predicted_clf[obj_indices]

    matched_cnt = len(target_clf_matched)
    hard_neg_cnt = int(matched_cnt * npr)

    if matched_cnt == 0:
        return torch.from_numpy(np.zeros(shape=(1,))), torch.from_numpy(np.zeros(shape=(1,))), 0

    target_loc_matched = target_loc[obj_indices]
    predicted_loc_matched = predicted_loc[obj_indices]

    regr_loss = smooth_l1_loss(predicted_loc_matched, target_loc_matched, reduction='sum')
    target_clf_matched_ids = target_clf_matched[:, :-1].argmax(dim=1)
    clf_loss_matched = cross_entropy(predicted_clf_matched[:, :-1], target_clf_matched_ids, reduction='sum')

    predicted_clf_mismatched = predicted_clf[noobj_indices]
    worst_errors_ids = get_hard_neg_ids(predicted_clf_mismatched, hard_neg_cnt)
    hard_neg_samples = predicted_clf_mismatched[worst_errors_ids]
    hard_neg_target = torch.from_numpy(np.ones(shape=(len(hard_neg_samples),), dtype=np.float32))
    hard_neg_target = hard_neg_target.to(hard_neg_samples.device)  # transfer tensor if use cuda
    clf_loss_mismatched = binary_cross_entropy(hard_neg_samples[:, -1], hard_neg_target, reduction='sum')

    return clf_loss_matched, clf_loss_mismatched, regr_loss, matched_cnt


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=False, reduce='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            bce_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-bce_loss)
        f_loss = self.alpha * (1-pt)**self.gamma * bce_loss

        if self.reduce == 'mean':
            return torch.mean(f_loss)
        elif self.reduce == 'sum':
            return torch.sum(f_loss)
        else:
            return f_loss


def head_loss_focal(target_clf, target_loc, predicted_clf, predicted_loc, coords=4, classes=21):

    # permute tensor b,c,y,x -> b,y,x,c
    predicted_clf = tensor_bcyx2byxc(predicted_clf)
    predicted_loc = tensor_bcyx2byxc(predicted_loc)
    target_clf = tensor_bcyx2byxc(target_clf)
    target_loc = tensor_bcyx2byxc(target_loc)

    # reshape tensors to b,y,x,anchor, coords
    predicted_loc = predicted_loc.reshape(-1, coords)
    target_loc = target_loc.reshape(-1, coords)
    predicted_clf = predicted_clf.reshape(-1, classes)
    target_clf = target_clf.reshape(-1, classes)

    # apply softmax to clf head
    predicted_clf = torch.nn.functional.softmax(predicted_clf, dim=1)

    # find all matched anchors
    obj_indices = (target_clf[:, -1] == 0)  # last probability = 0 <-> <No Onject> class
    noobj_indices = ~obj_indices
    focal_loss = FocalLoss(gamma=2, reduce='sum')

    # define all subsets by matching
    target_clf_pos = target_clf[obj_indices]
    predicted_clf_pos = predicted_clf[obj_indices]

    matched_cnt = len(target_clf_pos)

    if matched_cnt == 0:
        return torch.from_numpy(np.zeros(shape=(1,))), torch.from_numpy(np.zeros(shape=(1,))), \
               torch.from_numpy(np.zeros(shape=(1,))), 0, 0

    clf_loss_pos = focal_loss.forward(predicted_clf_pos, target_clf_pos)

    target_clf_neg = target_clf[noobj_indices]
    predicted_clf_neg = predicted_clf[noobj_indices]
    clf_loss_neg = focal_loss.forward(predicted_clf_neg, target_clf_neg)

    target_loc_pos = target_loc[obj_indices]
    predicted_loc_pos = predicted_loc[obj_indices]
    regr_loss = smooth_l1_loss(predicted_loc_pos, target_loc_pos, reduction='sum')

    return clf_loss_pos, clf_loss_neg, regr_loss, matched_cnt, len(predicted_clf)


def ssd_loss_focal(target, predicted, alpha=1., norm_type="total"):
    clf_loss_pos = torch.from_numpy(np.zeros(shape=(6,), dtype=np.float32))
    clf_loss_neg = torch.from_numpy(np.zeros(shape=(6,), dtype=np.float32))
    regr_loss = torch.from_numpy(np.zeros(shape=(6,), dtype=np.float32))
    pos_cnt = torch.from_numpy(np.zeros(shape=(6,), dtype=np.float32))
    total_cnt = torch.from_numpy(np.zeros(shape=(6,), dtype=np.float32))

    for i in range(6):
        clf_loss_pos[i], clf_loss_neg[i], regr_loss[i], pos_cnt[i], total_cnt[i] \
            = head_loss_focal(target[2 * i], target[2 * i + 1], predicted[2*i], predicted[2*i+1])

    if pos_cnt.sum():
        positive_count = pos_cnt.sum()
        if norm_type == "total":
            total_count = total_cnt.sum()
            return clf_loss_pos.sum() / total_count, clf_loss_neg.sum() / total_count, \
                   alpha * regr_loss.sum() / positive_count
        elif norm_type == "positive":
            return clf_loss_pos.sum() / positive_count, clf_loss_neg.sum() / positive_count, \
                   alpha * regr_loss.sum() / positive_count
    else:
        return torch.zeros(size=(1,)), torch.zeros(size=(1,)), torch.zeros(size=(1,))


class TestTrainingFunctions(unittest.TestCase):

    def test_get_hard_neg_ids(self):
        probs = [[0.2, 0.4, 0.1, 0.],
                 [0.1, 0.7, 0.1, 0.],
                 [0.05, 0.1, 0.03, 0.],
                 [0.1, 0.01, 0.08, 1],
                 [0., 0.4, 0.1, 0.],
                 [0., 0.4, 0.9, 0.],
                 [0., 0.4, 0.5, 0.],
                 [0., 0.99, 0.1, 0.]]
        tns = torch.from_numpy(np.asarray(probs).astype(np.float32))
        res = get_hard_neg_ids(tns, 3)
        self.assertEqual(set(res), set([7, 5, 1]))
        res = get_hard_neg_ids(tns, 1)
        self.assertEqual(set(res), set([7]))
        res = get_hard_neg_ids(tns, 4)
        self.assertEqual(set(res), set([7, 5, 1, 6]))
        return

    def test_hard_neg_ids_speed(self):
        test_data = 150000*[21*[random.random()]]
        test_tensor = torch.from_numpy(np.asarray(test_data))
        self.assertEqual(test_tensor.shape, (150000, 21))

        print("Start speed test")
        d = LogDuration()
        res = get_hard_neg_ids(test_tensor, 4)
        print("Highest prob: ", test_tensor[res])
        print("Speed test: ", d.get())

    def test_focal_loss(self):
        target = torch.from_numpy(np.ones(shape=(2, 8)).astype(np.float32))
        pred = torch.from_numpy(np.zeros(shape=(2, 8)).astype(np.float32))
        fl = FocalLoss(gamma=2)
        loss = fl.forward(pred, target)
        print(loss)
        return


if __name__ == "__main__":
    unittest.main()
