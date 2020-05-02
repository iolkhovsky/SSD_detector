from common_utils.bbox import *
from metrics.real_metrics import *


def mean_average_precision(predicted, gtruth):
    """
    predicted / grtuth - list of list of detections:
    detection (pascal_voc.descriptors)
    x, y, w, h
    p, id, label
    """
    out = dict()  # keys: "map" and ap for each class name

    # 1. For each class we have to build 2 lists - target and predicted
    for predicted_objects, target_objects in zip(predicted, gtruth):
        image_result = {}  # results for current pair (target/pred) in image
        for target in target_objects:  # for each target box
            for pred in predicted_objects:  # looking for response
                if pred.id == target.id:
                    pred_box = BBox(coords=(pred.x, pred.y, pred.w, pred.h))
                    target_box = BBox(coords=(target.x, target.y, target.w, target.h))
                    if target_box.get_iou(pred_box) > 0.5:
                        obj_class = target.label
                        res = (pred_box, pred.p, 1.0)
                        if obj_class in image_result.keys():
                            image_result[obj_class].append(res)
                        else:
                            image_result[obj_class] = [res]
        for pred in predicted_objects:  # for each predicted box
            pred_box = BBox(coords=(pred.x, pred.y, pred.w, pred.h))
            pos_cnt = 0
            for target in target_objects:
                if target.id == pred.id:
                    target_box = BBox(coords=(target.x, target.y, target.w, target.h))
                    if pred_box.get_iou(target_box) > 0.5:
                        pos_cnt += 1
                        break
            #  if noone target box has iou with cur pred box
            if pos_cnt == 0:
                obj_class = pred.label
                res = (pred_box, pred.p, 0.0)
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
    cls_cnt = 0
    map_val = 0.
    for cls in out.keys():
        y = out[cls]["y"]
        y_ = out[cls]["y_"]
        ap = get_voc_ap(y, y_)
        out[cls] = ap
        map_val += ap
        cls_cnt += 1
    if cls_cnt > 0:
        out["map"] = map_val / cls_cnt
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