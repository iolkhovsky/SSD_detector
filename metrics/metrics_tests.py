from metrics.detection_metrics import *


def test_accuracy():
    print("Test accuracy function: ")
    y = [1, 0, 0, 1, 0, 0]
    y_ = [1, 1, 0, 1, 0, 1]
    target = 4./6.
    eps = 1e-5
    if abs(accuracy(y, y_) - target) < eps:
        print("Ok")
    else:
        print("Failed! Accuracy: ", accuracy(y, y_), " Target: ", target)
        print("Input: y=", y, " y_=", y_)


def test_precision():
    print("Test precision function: ")
    y = [1, 0, 0, 1, 1, 0]
    y_ = [1, 1, 0, 1, 0, 1]
    target = 2./3.
    eps = 1e-5
    if abs(precision(y, y_) - target) < eps:
        print("Ok")
    else:
        print("Failed! Precision: ", precision(y, y_), " Target: ", target)
        print("Input: y=", y, " y_=", y_)


def test_recall():
    print("Test recall function: ")
    y = [1, 0, 0, 1, 1, 0]
    y_ = [1, 1, 0, 1, 0, 1]
    target = 2./4.
    eps = 1e-5
    if abs(recall(y, y_) - target) < eps:
        print("Ok")
    else:
        print("Failed! Recall: ", recall(y, y_), " Target: ", target)
        print("Input: y=", y, " y_=", y_)


def test_f1_score():
    print("Test f1score function: ")
    y = [1, 0, 0, 1, 1, 0]
    y_ = [1, 1, 0, 1, 0, 1]
    target_rec = 2./4.
    target_pr = 2./3.
    target_f1 = 2*target_pr*target_rec/(target_pr + target_rec)
    eps = 1e-5
    if abs(f1_score(y, y_) - target_f1) < eps:
        print("Ok")
    else:
        print("Failed! F1: ", recall(y, y_), " Target: ", target_f1)
        print("Input: y=", y, " y_=", y_)


def test_prc_curve():
    y = [0.14, 0.23, 0.39, 0.54, 0.73, 0.9]
    y_ = [0., 1., 0., 0., 1., 1.]
    show_prc(y, y_)
    voc_ap = get_voc_ap(y, y_)
    print("VOC AP: ", voc_ap)


def test_roc_curve():
    y = [0.14, 0.23, 0.39, 0.54, 0.73, 0.9]
    y_ = [0., 1., 0., 0., 1., 1.]
    show_roc(y, y_)
    _roc_auc = roc_auc(y, y_)
    print("ROC AUC: ", _roc_auc)


def test_map():
    print("Testing mAP function")
    pred = [[{"bbox": (10, 20, 30, 40), "conf": 0.7, "class": "dog", "class_id": 0, "prob": 0.5},
             {"bbox": (100, 200, 30, 30), "conf": 0.4, "class": "dog", "class_id": 0, "prob": 0.5}],
            [{"bbox": (15, 25, 30, 40), "conf": 0.9, "class": "person", "class_id": 1, "prob": 0.5},
             {"bbox": (1, 2, 3, 4), "conf": 0.3, "class": "person", "class_id": 1, "prob": 0.5},
             {"bbox": (1, 2, 3, 4), "conf": 0.4, "class": "chair", "class_id": 2, "prob": 0.5}]]
    target = [[{"bbox": (9, 19, 30, 40), "conf": 1.0, "class": "dog", "class_id": 0, "prob": 0.5}],
              [{"bbox": (14, 24, 29, 39), "conf": 1.0, "class": "person", "class_id": 1, "prob": 0.5}]]
    res = mean_average_precision(pred, target)
    print(res)