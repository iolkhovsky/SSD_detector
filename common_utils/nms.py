from collections import defaultdict

from common_utils.bbox import BBox


def non_max_suppression(detections, iou=0.5):
    out = []
    for img_detections in detections:
        out_img = []
        det_by_classes = defaultdict(list)
        img_detections = sorted(img_detections, key=lambda x: -x.p)
        for det in img_detections:
            box = BBox((det.x, det.y, det.w, det.h))
            ok = True
            for existing_det in det_by_classes[det.id]:
                exist_box = BBox((existing_det.x, existing_det.y, existing_det.w, existing_det.h))
                if exist_box.get_iou(box) >= iou:
                    ok = False
                    break
            if ok:
                det_by_classes[det.id].append(det)
                out_img.append(det)
        out.append(out_img)
    return out
