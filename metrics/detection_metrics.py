from collections import defaultdict
from math import isnan
from sklearn.metrics import average_precision_score

from common_utils.bbox import *
from metrics.real_metrics import *


def divide_by_class(predicted, gtruth):
    assert len(predicted) == len(gtruth)
    image_cnt = len(predicted)
    labels = set()
    gt_by_class, predicted_by_class = {}, {}
    for img_id in range(image_cnt):
        for pr_detection in predicted[img_id]:
            labels.add(pr_detection.label)
            if pr_detection.label not in predicted_by_class.keys():
                predicted_by_class[pr_detection.label] = []
                for i in range(image_cnt):
                    predicted_by_class[pr_detection.label].append([])
            predicted_by_class[pr_detection.label][img_id].append(pr_detection)
        for gt_detection in gtruth[img_id]:
            labels.add(gt_detection.label)
            if gt_detection.label not in gt_by_class.keys():
                gt_by_class[gt_detection.label] = []
                for i in range(image_cnt):
                    gt_by_class[gt_detection.label].append([])
            gt_by_class[gt_detection.label][img_id].append(gt_detection)
    return gt_by_class, predicted_by_class, labels


def get_score_label_sample(gt_detections, pr_detections, iou_threshold=0.5):
    out = []
    if len(gt_detections) == 0:
        for det in pr_detections:
            out.append((det.p, 0))
        return out
    if len(pr_detections) == 0:
        for _ in gt_detections:
            out.append((0., 1))

    iou_matrix = np.zeros(shape=(len(gt_detections), len(pr_detections)), dtype=np.float)
    for gt_idx, gt in enumerate(gt_detections):
        for pr_idx, pr in enumerate(pr_detections):
            pred_box = BBox(coords=(pr.x, pr.y, pr.w, pr.h))
            target_box = BBox(coords=(gt.x, gt.y, gt.w, gt.h))
            iou_matrix[gt_idx, pr_idx] = pred_box.get_iou(target_box)
    match_matrix = np.greater_equal(iou_matrix, iou_threshold)
    correct_preds = np.any(match_matrix, axis=0)
    skipped_gt = np.all(np.logical_not(match_matrix), axis=1)
    for pred_idx, pred in enumerate(pr_detections):
        out.append((pred.p, int(correct_preds[pred_idx])))
    for gt_idx, gt in enumerate(gt_detections):
        if skipped_gt[gt_idx]:
            out.append((0., 1))
    return out


def compile_detections_sample(gt, pred, iou_threshold=0.5):
    assert len(gt) == len(pred)
    out = []
    for img_idx, frame_gt_detections in enumerate(gt):
        frame_pr_detections = pred[img_idx]
        out.extend(get_score_label_sample(frame_gt_detections, frame_pr_detections, iou_threshold=iou_threshold))
    return out


def mean_average_precision(predicted, gtruth, iou_threshold=0.5):
    """
    predicted / grtuth - list of list of detections:
    detection (pascal_voc.descriptors)
    x, y, w, h
    p, id, label
    """
    out = dict()
    gt_by_class, predicted_by_class, labels = divide_by_class(predicted, gtruth)
    empty_detections = []
    for i in range(len(predicted)):
        empty_detections.append([])
    for label in labels:
        gt_detections = gt_by_class[label] if label in gt_by_class.keys() else empty_detections
        pred_detections = predicted_by_class[label] if label in predicted_by_class.keys() else empty_detections
        score_label_sample = compile_detections_sample(gt=gt_detections, pred=pred_detections,
                                                       iou_threshold=iou_threshold)
        score_label_sample = sorted(score_label_sample, key=lambda tup: tup[0])
        y_scores, y_true = zip(*score_label_sample)
        ap = average_precision_score(y_true=y_true, y_score=y_scores)
        out[label] = 0. if isnan(ap) else ap
    out["mAP"] = sum(out.values()) / float(len(out.values()))
    return out


def get_optimal_threshold(predicted, gtruth):
    """
    predicted / grtuth - list of list of dicts:
    {"bbox": - tuple of x,y,w,h
    "conf": - confidence score
    "class": - label of the class
    "class_id" - id of the class
    "prob" - probability for the class}
    external list's content - description for definite image file
    internal list's content - decsriptions of objects on the image
    dict - description of object
    """
    out = dict()  # keys: "map" and ap for each class name

    # 1. For each class we have to build 2 lists - target and predicted
    for predicted_objects, target_objects in zip(predicted, gtruth):
        image_result = {}  # results for current pair (target/pred) in image
        target_box, pred_box = BBox(), BBox()
        for target in target_objects:  # for each target box
            target_box.set_abs(target["bbox"])
            for pred in predicted_objects:  # looking for response
                if pred["class_id"] == target["class_id"]:
                    pred_box.set_abs(pred["bbox"])
                    if target_box.iou(pred_box) > 0.5:
                        obj_class = target["class"]
                        res = (pred_box, pred["conf"], 1.0)
                        if obj_class in image_result.keys():
                            image_result[obj_class].append(res)
                        else:
                            image_result[obj_class] = [res]
        for pred in predicted_objects:  # for each predicted box
            pred_box.set_abs((pred["bbox"]))
            pos_cnt = 0
            for target in target_objects:
                if target["class_id"] == pred["class_id"]:
                    target_box.set_abs(target["bbox"])
                    if pred_box.iou(target_box) > 0.5:
                        pos_cnt += 1
                        break
            #  if noone target box has iou with cur pred box
            if pos_cnt == 0:
                obj_class = pred["class"]
                res = (pred_box, pred["conf"], 0.0)
                if obj_class in image_result.keys():
                    image_result[obj_class].append(res)
                else:
                    image_result[obj_class] = [res]

        # now we should convert image result to out dict
        for classname in image_result.keys():
            for sample in image_result[classname]:
                bbox, conf, label = sample
                if classname in out.keys():
                    out[classname]["y"].append(conf)
                    out[classname]["y_"].append(label)
                else:
                    out[classname] = dict()
                    out[classname]["y"] = [conf]
                    out[classname]["y_"] = [label]

    # now for each class we should compute AP
    common_y = []
    common_y_ = []

    for cls in out.keys():
        y = out[cls]["y"]
        y_ = out[cls]["y_"]
        common_y.extend(y)
        common_y_.extend(y_)

    roc_with_thresh = roc_curve_with_thresh(common_y, common_y_)
    show_roc(common_y, common_y_)

    # looking for closest point
    best_point = None
    for point in roc_with_thresh:
        fpr, tpr, thresh = point
        dist = np.sqrt(np.power(1. - tpr, 2) + np.power(fpr, 2))
        if best_point is None:
            best_point = point, dist
        else:
            if dist < best_point[1]:
                best_point = point, dist
    best_point = best_point[0]
    print("Best point (fpr, tpr, thresh): ", best_point)
    return best_point[2]