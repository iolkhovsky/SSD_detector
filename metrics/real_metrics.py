import numpy as np
from metrics.binary_metrics import *
from matplotlib import pyplot as plt


# PRECISION RECALL CURVE ANALYSIS ******************************************************

# get curve
def pr_curve(y, y_):  # precision recall characteristic
    # sort y (prediction) and appropriate y_ (target)  in order low -> high
    y, y_ = zip(*sorted(zip(y, y_)))
    y, y_ = list(y), list(y_)
    # sequentally decrease threshold and generate point of pr curve
    threshold = 1.
    points = [(0, 0)]
    sz = len(y)
    for i in range(len(y)):
        threshold = y[sz - 1 - i]
        y_bin = np.asarray(y) >= threshold
        pr = precision(y_bin, y_)
        rec = recall(y_bin, y_)
        point = (rec, pr)
        points.append(point)
    return points


# get recrified (non increas) curve
def rectified_pr_curve(prc):
    # reverse prc
    prc = prc[::-1]
    rect_points = []
    for p in prc:
        if len(rect_points):
            last_rec, last_pr = rect_points[-1]
            cur_rec, cur_pr = p
            if cur_pr > last_pr:
                rect_points.append(p)
            else:
                rect_points.append((cur_rec, last_pr))
        else:
            rect_points.append(p)
    return rect_points[::-1]


def prc_auc(y, y_):
    prc = pr_curve(y, y_)
    rect_prc = rectified_pr_curve(prc)
    rect_prc = drop_same_pos_prc(rect_prc)
    rec_coords, pr_coords = zip(*rect_prc)

    area = 0.
    for idx, ref_rec in enumerate(rec_coords):
        if idx > 0:
            area += (rec_coords[idx] - rec_coords[idx - 1]) * pr_coords[idx]
    return area


def get_voc_ap(y, y_):
    prc = pr_curve(y, y_)
    rect_prc = rectified_pr_curve(prc)
    rect_prc = drop_same_pos_prc(rect_prc)
    rec_coords, pr_coords = zip(*rect_prc)

    rec_step = 0.1
    ref_recall = [rec_step * i for i in range(11)]

    area = 0.

    for idx, ref_rec in enumerate(ref_recall):
        if idx > 0:
            left, right = get_left_right(ref_rec, rec_coords)
            if left is not None and right is not None:
                area += (right - left) * rec_step * pr_coords[right]
    return area


# drop points on the same recall (x - coord)
def drop_same_pos_prc(curve):
    out = []
    for idx, point in enumerate(curve):
        if len(out):
            cur_rec, cur_pr = point
            prev_rec, prev_pr = out[-1]
            if cur_rec != prev_rec:
                out.append(point)
        else:
            out.append(point)
    return out


# get left and right index for 1d interpolation
def get_left_right(ref, arr):
    left, right = None, None
    for i in range(len(arr)):
        if left is not None:
            if arr[i] < ref:
                left = i
        else:
            left = i
        if right is None:
            if arr[i] >= ref:
                right = i
                break
    return left, right


def show_prc(y, y_):
    curve = pr_curve(y, y_)
    rect_curv = rectified_pr_curve(curve)
    rec_coords, pr_coords = zip(*curve)
    rect_rec_coords, rect_pr_coords = zip(*rect_curv)

    fig = plt.figure()
    ax = fig.gca()
    ax.set_xticks(np.arange(0, 1, 0.1))
    ax.set_yticks(np.arange(0, 1., 0.1))
    plt.plot(rec_coords, pr_coords, 'b')
    plt.plot(rect_rec_coords, rect_pr_coords, 'r')
    plt.scatter(rec_coords, pr_coords)
    plt.scatter(rect_rec_coords, rect_pr_coords)
    plt.show()
    pass

# RECEIVER OPERATION CURVE ANALYSIS **********************************************************************


def roc_curve(y, y_):  # receiver operating cheracteristic
    # sort y (prediction) and appropriate y_ (target)  in order low -> high
    y, y_ = zip(*sorted(zip(y, y_)))
    y, y_ = list(y), list(y_)
    # sequentally decrease threshold and generate point of pr curve
    threshold = 1.
    points = [(0, 0)]
    sz = len(y)
    for i in range(len(y)):
        threshold = y[sz - 1 - i]
        y_bin = np.asarray(y) >= threshold
        tpr = tp_rate(y_bin, y_)
        fpr = fp_rate(y_bin, y_)
        point = (fpr, tpr)
        points.append(point)
    return points


def roc_curve_with_thresh(y, y_):  # receiver operating cheracteristic
    # sort y (prediction) and appropriate y_ (target)  in order low -> high
    y, y_ = zip(*sorted(zip(y, y_)))
    y, y_ = list(y), list(y_)
    # sequentally decrease threshold and generate point of pr curve
    threshold = 1.
    points = [(0, 0, 1.0)]
    sz = len(y)
    for i in range(len(y)):
        threshold = y[sz - 1 - i]
        y_bin = np.asarray(y) >= threshold
        tpr = tp_rate(y_bin, y_)
        fpr = fp_rate(y_bin, y_)
        point = (fpr, tpr, threshold)
        points.append(point)
    return points


def roc_auc(y, y_):
    roc = roc_curve(y, y_)
    fpr_coords, tpr_coords = zip(*roc)
    area = 0.
    for idx, fpr_rec in enumerate(fpr_coords):
        if idx > 0:
            area += (fpr_coords[idx] - fpr_coords[idx - 1]) * tpr_coords[idx - 1]
    return area


def show_roc(y, y_):
    curve = roc_curve(y, y_)
    fpr_coords, tpr_coords = zip(*curve)

    fig = plt.figure()
    ax = fig.gca()
    ax.set_xticks(np.arange(0, 1, 0.1))
    ax.set_yticks(np.arange(0, 1., 0.1))
    plt.plot(fpr_coords, tpr_coords, 'b')
    plt.scatter(fpr_coords, tpr_coords)
    plt.title("ROC curve")
    plt.savefig("roc_curve.png")
    plt.show()
    pass
