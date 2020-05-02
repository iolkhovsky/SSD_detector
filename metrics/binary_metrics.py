import numpy as np


def accuracy(y, y_):
    """
    y - predicted label (0/1)
    y_ - target label (0/1)
    """
    if len(y) != len(y_):
        raise ValueError("Wrong input data")
    y = np.asarray(y, dtype=np.int)
    y_ = np.asarray(y_, dtype=np.int)
    if len(y.shape) > 1 or len(y_.shape) > 1:
        raise ValueError("Wrong shape of in data")
    match = np.asarray(y == y_, dtype=np.float32)
    return match.mean()


def precision(y, y_):
    """
    y - predicted label (0/1)
    y_ - target label (0/1)
    """
    if len(y) != len(y_):
        raise ValueError("Wrong input data")
    y = np.asarray(y, dtype=np.int)
    y_ = np.asarray(y_, dtype=np.int)
    if len(y.shape) > 1 or len(y_.shape) > 1:
        raise ValueError("Wrong shape of in data")
    ''' Pr = TP / (TP+FP)'''
    TP = np.sum(np.logical_and(np.equal(y, y_), y_))
    FP = np.sum(np.logical_and(y == 1., y_ == 0))
    if TP + FP == 0:
        return 0
    return TP / (TP + FP)


def recall(y, y_):
    """
    y - predicted label (0/1)
    y_ - target label (0/1)
    """
    if len(y) != len(y_):
        raise ValueError("Wrong input data")
    y = np.asarray(y, dtype=np.int)
    y_ = np.asarray(y_, dtype=np.int)
    if len(y.shape) > 1 or len(y_.shape) > 1:
        raise ValueError("Wrong shape of in data")
    ''' Pr = TP / (TP+FN)'''
    TP = np.sum(np.logical_and(np.equal(y, y_), y_))
    FN = np.sum(np.logical_and(y == 0., y_ == 1))
    if TP + FN == 0:
        return 0
    return TP / (TP + FN)


def f1_score(y, y_):
    pr = precision(y, y_)
    rec = recall(y, y_)
    return 2*pr*rec/(pr + rec)


def tp_rate(y, y_):
    """
    TPR = TP/(TP + FN) - 1 to 1 as Recall
    """
    if len(y) != len(y_):
        raise ValueError("Wrong input data")
    y = np.asarray(y, dtype=np.int)
    y_ = np.asarray(y_, dtype=np.int)
    if len(y.shape) > 1 or len(y_.shape) > 1:
        raise ValueError("Wrong shape of in data")
    TP = np.sum(np.logical_and(np.equal(y, y_), y_))
    FN = np.sum(np.logical_and(y == 0., y_ == 1))
    if TP + FN == 0:
        return 0
    return TP / (TP + FN)


def fp_rate(y, y_):
    """
    FPR = FP/(FP + TN)
    """
    if len(y) != len(y_):
        raise ValueError("Wrong input data")
    y = np.asarray(y, dtype=np.int)
    y_ = np.asarray(y_, dtype=np.int)
    if len(y.shape) > 1 or len(y_.shape) > 1:
        raise ValueError("Wrong shape of in data")
    FP = np.sum(np.logical_and(y == 1, y_ == 0))
    TN = np.sum(np.logical_and(y == 0, y_ == 0))
    if FP + TN == 0:
        return 0
    return FP / (FP + TN)
