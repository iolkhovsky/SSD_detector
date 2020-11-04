from numpy import sqrt, log, exp
import cv2

from pascal_voc.descriptors import DetectionDescriptor
from common_utils.bbox import BBox
from ssd.utils import get_scale_for_fmap
from common_utils.tensor_transform import *
from pascal_voc.descriptors import ObjectDescriptor
from ssd.constraints import *


class PriorsGenerator:

    def __init__(self, imgsz, fmap_cnt, prior_ratio_list=DEFAULT_HW_RATIOS):
        assert len(list(imgsz)) == 2
        assert type(fmap_cnt) == int
        assert len(list(prior_ratio_list)) >= 1
        self.imgsz = imgsz
        self.fmap_cnt = fmap_cnt
        self.priors = prior_ratio_list

        self.scales = [get_scale_for_fmap(i, self.fmap_cnt) for i in range(self.fmap_cnt)]
        return

    def get_box(self, fmap_idx, prior):
        assert (fmap_idx >= 0) and (fmap_idx < self.fmap_cnt)
        assert (prior >= 0) and (prior < len(self.priors))
        scale = self.scales[fmap_idx]
        if prior == len(self.priors) - 1:  # if the last prior with aspect ration 1
            scale = self.scales[fmap_idx] * 1.0  # 1.0 as image size for the last map
        anchor_w = scale * sqrt(self.priors[prior])
        anchor_h = scale / sqrt(self.priors[prior])
        anchor_w *= self.imgsz[0]
        anchor_h *= self.imgsz[1]
        return BBox((0., 0., anchor_w, anchor_h))

    def get_cnt(self):
        return len(self.priors)

    def __len__(self):
        return len(self.priors)


class SSDBoxCodec:

    def __init__(self, imgsz, feat_maps_sz, classes_cnt=20, coords_cnt=4):
        self.priors = PriorsGenerator(imgsz, len(feat_maps_sz))
        self.feat_maps_sz = feat_maps_sz
        self.imgsz = imgsz
        self.full_classes_cnt = classes_cnt + 1
        self.coords_cnt = coords_cnt
        return

    def encode(self, src_box, iou_threshold=0.5):
        out = []
        for lvl_idx, fmap_size in enumerate(self.feat_maps_sz):
            for y_idx in range(fmap_size[1]):
                for x_idx in range(fmap_size[0]):
                    for anchor_idx in range(self.priors.get_cnt()):
                        cur_anchor = self.decode(lvl_idx, y_idx, x_idx, anchor_idx)
                        iou = cur_anchor.get_iou(src_box)
                        if iou >= iou_threshold:
                            out.append((lvl_idx, y_idx, x_idx, anchor_idx))
        return out

    def decode(self, lvl_idx, y_idx, x_idx, anchor_idx, resolve_negatives=True):
        base_prior = self.priors.get_box(lvl_idx, anchor_idx)

        fmap_sz = self.feat_maps_sz[lvl_idx]
        cell_sz_x, cell_sz_y = 1./fmap_sz[0], 1./fmap_sz[1]

        xc = (x_idx + 0.5) * cell_sz_x
        yc = (y_idx + 0.5) * cell_sz_x

        xc *= self.imgsz[0]
        yc *= self.imgsz[1]

        base_prior.x = int(xc - base_prior.w * 0.5)
        base_prior.y = int(yc - base_prior.h * 0.5)
        base_prior.w = int(base_prior.w)
        base_prior.h = int(base_prior.h)
        if resolve_negatives:
            base_prior.make_valid(self.imgsz)
        return base_prior

    def get_classes_cnt(self):
        return self.full_classes_cnt

    def get_coords_cnt(self):
        return self.coords_cnt

    def get_priors_cnt(self):
        return len(self.priors)

    def get_fmap_size(self, idx):
        return self.feat_maps_sz[idx]

    def get_levels_cnt(self):
        return len(self.feat_maps_sz)

    @staticmethod
    def get_norm_coords(ground_truth, def_anchor):
        gt_cx = ground_truth.x + 0.5 * ground_truth.w
        gt_cy = ground_truth.y + 0.5 * ground_truth.h
        anchor_cx = def_anchor.x + 0.5 * def_anchor.w
        anchor_cy = def_anchor.y + 0.5 * def_anchor.h
        norm_x = (gt_cx - anchor_cx) / def_anchor.w
        norm_y = (gt_cy - anchor_cy) / def_anchor.h
        norm_w = log(ground_truth.w / def_anchor.w)
        norm_h = log(ground_truth.h / def_anchor.h)
        return norm_x, norm_y, norm_w, norm_h

    @staticmethod
    def get_abs_coords(norm_coord, def_anchor):
        anchor_cx = def_anchor.x + 0.5 * def_anchor.w
        anchor_cy = def_anchor.y + 0.5 * def_anchor.h
        cx = norm_coord.x * def_anchor.w + anchor_cx
        cy = norm_coord.y * def_anchor.h + anchor_cy
        abs_w = exp(norm_coord.w) * def_anchor.w
        abs_h = exp(norm_coord.h) * def_anchor.h
        abs_x = cx - 0.5 * abs_w
        abs_y = cy - 0.5 * abs_h
        return abs_x, abs_y, abs_w, abs_h


class SSDCodec:

    def __init__(self, box_codec):
        self.box_codec = box_codec
        pass

    @staticmethod
    def set_noobj_prob(tns, pr_cnt, cls_cnt):
        for y_coord in range(tns.shape[0]):
            for x_coord in range(tns.shape[1]):
                for anchor_id in range(pr_cnt):
                    tns[y_coord, x_coord, anchor_id * cls_cnt + cls_cnt - 1] = 1
        return

    def encode(self, sample_descriptor):
        classes_cnt = self.box_codec.get_classes_cnt()
        coords_per_box = self.box_codec.get_coords_cnt()
        priors_cnt = self.box_codec.get_priors_cnt()

        objects = sample_descriptor.objects
        scaled_objects = []
        imgsz = sample_descriptor.img_size  # c, y, x

        kx, ky = float(self.box_codec.imgsz[0]) / imgsz[2], float(self.box_codec.imgsz[1]) / imgsz[1]

        for obj in objects:
            sdesc = ObjectDescriptor()
            sdesc.x, sdesc.w = obj.x * kx, obj.w * kx
            sdesc.y, sdesc.h = obj.y * ky, obj.h * ky
            sdesc.class_id = obj.class_id
            sdesc.class_label = obj.class_label
            scaled_objects.append(sdesc)

        clf_tensors = []
        loc_tensors = []
        levels_cnt = self.box_codec.get_levels_cnt()

        for i in range(levels_cnt):
            fmap_sz = self.box_codec.get_fmap_size(i)
            classifier = torch.from_numpy(np.zeros(shape=(fmap_sz[0], fmap_sz[1], classes_cnt * priors_cnt),
                                                   dtype=np.float32))
            regressor = torch.from_numpy(np.zeros(shape=(fmap_sz[0], fmap_sz[1], coords_per_box * priors_cnt),
                                                  dtype=np.float32))
            clf_tensors.append(classifier)
            loc_tensors.append(regressor)

        # set all clf outputs as default - no object
        for i in range(levels_cnt):
            self.set_noobj_prob(clf_tensors[i], priors_cnt, classes_cnt)

        for target_object in scaled_objects:
            groundtruth_box = BBox((target_object.x, target_object.y, target_object.w, target_object.h))
            anchors = self.box_codec.encode(groundtruth_box)
            for lvl_idx, y_position, x_position, anchor_idx in anchors:
                default_box = self.box_codec.decode(lvl_idx, y_position, x_position, anchor_idx)
                x_norm, y_norm, w_norm, h_norm = SSDBoxCodec.get_norm_coords(groundtruth_box, default_box)
                loc_tensors[lvl_idx][y_position, x_position, anchor_idx * coords_per_box + X_OFFSET] = x_norm
                loc_tensors[lvl_idx][y_position, x_position, anchor_idx * coords_per_box + Y_OFFSET] = y_norm
                loc_tensors[lvl_idx][y_position, x_position, anchor_idx * coords_per_box + W_OFFSET] = w_norm
                loc_tensors[lvl_idx][y_position, x_position, anchor_idx * coords_per_box + H_OFFSET] = h_norm
                clf_tensors[lvl_idx][y_position, x_position, anchor_idx * classes_cnt + target_object.class_id] = 1
                clf_tensors[lvl_idx][y_position, x_position, anchor_idx * classes_cnt + classes_cnt - 1] = 0

        out = []
        for i in range(levels_cnt):
            out.append(tensor_yxc2cyx(clf_tensors[i]))
            out.append(tensor_yxc2cyx(loc_tensors[i]))

        return tuple(out)

    def decode(self, head_tensors, id2class=None, prediction=False, threshold=None):
        out = []

        clf_0, reg_0, clf_1, reg_1, clf_2, reg_2, clf_3, reg_3, clf_4, reg_4, clf_5, reg_5 = head_tensors
        clf = [clf_0.detach().numpy(), clf_1.detach().numpy(), clf_2.detach().numpy(), clf_3.detach().numpy(),
               clf_4.detach().numpy(), clf_5.detach().numpy()]
        reg = [reg_0.detach().numpy(), reg_1.detach().numpy(), reg_2.detach().numpy(), reg_3.detach().numpy(),
               reg_4.detach().numpy(), reg_5.detach().numpy()]

        for i in range(len(clf)):
            clf[i] = array_cyx2yxc(clf[i])
            reg[i] = array_cyx2yxc(reg[i])

        for level in range(6):
            fmap_size = self.box_codec.get_fmap_size(level)
            for y_position in range(fmap_size[0]):
                for x_position in range(fmap_size[1]):
                    priors_cnt = self.box_codec.get_priors_cnt()
                    classes_cnt = self.box_codec.get_classes_cnt()

                    for anchor_id in range(priors_cnt):
                        anchor_offset = anchor_id * classes_cnt
                        probs = clf[level][y_position, x_position][anchor_offset: anchor_offset + classes_cnt]
                        det = DetectionDescriptor()

                        if prediction:  # inference: detection is the most prob class
                            # probs_smax = custom_softmax(probs)
                            probs_smax = torch.nn.functional.softmax(torch.from_numpy(probs), dim=0).numpy()
                            det.id = np.argmax(probs_smax)
                            det.p = probs_smax[det.id]
                            if det.id != classes_cnt - 1:

                                if threshold:
                                    if det.p < threshold:
                                        continue  # ignore this detection

                                if id2class:
                                    det.label = id2class[det.id]
                                default_box = self.box_codec.decode(level, y_position, x_position, anchor_id)
                                norm_x = reg[level][
                                    y_position, x_position, anchor_id * self.box_codec.get_coords_cnt() + X_OFFSET]
                                norm_y = reg[level][
                                    y_position, x_position, anchor_id * self.box_codec.get_coords_cnt() + Y_OFFSET]
                                norm_w = reg[level][
                                    y_position, x_position, anchor_id * self.box_codec.get_coords_cnt() + W_OFFSET]
                                norm_h = reg[level][
                                    y_position, x_position, anchor_id * self.box_codec.get_coords_cnt() + H_OFFSET]
                                x, y, w, h = SSDBoxCodec.get_abs_coords(BBox((norm_x, norm_y, norm_w, norm_h)),
                                                                        default_box)
                                det.x, det.y, det.w, det.h = x, y, w, h
                                out.append(det)
                        elif probs[-1] == 0:
                            det.p = np.max(probs[:-1])
                            det.id = np.argmax(probs[:-1])
                            if id2class:
                                det.label = id2class[det.id]
                            default_box = self.box_codec.decode(level, y_position, x_position, anchor_id)
                            norm_x = reg[level][y_position, x_position, anchor_id * self.box_codec.get_coords_cnt() +
                                                X_OFFSET]
                            norm_y = reg[level][y_position, x_position, anchor_id * self.box_codec.get_coords_cnt() +
                                                Y_OFFSET]
                            norm_w = reg[level][y_position, x_position, anchor_id * self.box_codec.get_coords_cnt() +
                                                W_OFFSET]
                            norm_h = reg[level][y_position, x_position, anchor_id * self.box_codec.get_coords_cnt() +
                                                H_OFFSET]
                            x, y, w, h = SSDBoxCodec.get_abs_coords(BBox((norm_x, norm_y, norm_w, norm_h)), default_box)
                            det.x, det.y, det.w, det.h = x, y, w, h
                            out.append(det)
        return out


if __name__ == "__main__":
    codec = SSDBoxCodec((300, 300), [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)])
    # for level in range(6):
    #     for prior in range(6):
    #         for y_pos in range(codec.feat_maps_sz[level][1]):
    #             for x_pos in range(codec.feat_maps_sz[level][0]):
    #                 image = np.zeros(shape=(300, 300, 3), dtype=np.uint8)
    #                 box = codec.decode(level, anchor_idx=prior, y_idx=y_pos, x_idx=x_pos)
    #                 image = cv2.rectangle(img=image, pt1=(box.x, box.y), pt2=(box.x + box.w, box.y + box.h),
    #                                       color=[255, 0, 0], thickness=1)
    #                 cv2.imshow("Level"+str(level), image)
    #                 # cv2.waitKey(1)

    image = np.zeros(shape=(300, 300, 3), dtype=np.uint8)
    ref_box = BBox(coords=(100, 60, 40, 20))
    anchors_position = codec.encode(ref_box, 0.5)
    for anchor in anchors_position:
        level_idx, y_pos, x_pos, prior_idx = anchor
        box = codec.decode(lvl_idx=level_idx, y_idx=y_pos, x_idx=x_pos, anchor_idx=prior_idx)
        image = cv2.rectangle(img=image, pt1=(box.x, box.y), pt2=(box.x + box.w, box.y + box.h),
                              color=[255, 0, 0], thickness=1)
    image = cv2.rectangle(img=image, pt1=(ref_box.x, ref_box.y), pt2=(ref_box.x + ref_box.w, ref_box.y + ref_box.h),
                          color=[0, 0, 255], thickness=1)

    cv2.imshow("EncodeDecode", image)
    cv2.waitKey(100)

    gt_box = BBox(coords=(100, 60, 40, 50))
    anchor_box = BBox(coords=(8, 17, 50, 52))
    xn, yn, wn, hn = SSDBoxCodec.get_norm_coords(gt_box, anchor_box)
    norm_box = BBox(coords=(xn, yn, wn, hn))
    x, y, w, h = SSDBoxCodec.get_abs_coords(norm_box, anchor_box)

    print("Response", input("Completed. Input any symbol"))



