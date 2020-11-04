import cv2

from common_utils.tensor_transform import *
from common_utils.visualization import show_image_cv, draw_object_mark
from common_utils.logger import Logger
from common_utils.cache import Cache
from pascal_voc.voc_index import VocIndex
from ssd.ssd_codec import SSDCodec, SSDBoxCodec
from ssd.ssd_inference import visualize_inference
from ssd.constraints import *


class VocDetectionSSD:

    def __init__(self, root, target_shape=DEFAULT_TARGET_SIZE, feature_maps_shape=DEFAULT_FMAP_SIZES, logger=print,
                 cache=0, sz_limit=None):
        self.target_size = target_shape
        self.fmaps_size = feature_maps_shape
        self.index = VocIndex(root_path=root, index_name=root, size_limit=sz_limit)
        self.box_codec = SSDBoxCodec(self.target_size, self.fmaps_size)
        self.codec = SSDCodec(self.box_codec)
        self.cache = Cache((cache / 100.) * len(self.index))
        self.logger = logger
        if type(logger) == Logger:
            logger("Loaded Index ", self.index, caller="VocDetectionSSD")
        else:
            logger("Loaded Index ", self.index)
        return

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return [self[i] for i in range(*item.indices(len(self)))]
        elif isinstance(item, int):
            return self.__make_sample(item)
        else:
            raise ValueError("Invalid index(-ices) to __get_item__ method")

    def __str__(self):
        return "Pascal VOC dataset for detection from " + str(self.index)

    def __make_sample(self, idx):
        # in image
        sample_desc = self.index[idx]
        path = sample_desc.abs_path
        image = cv2.imread(path)
        image = cv2.resize(image, (self.target_size[0], self.target_size[1]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # reverse bgr to rgb
        image = array_yxc2cyx(image)  # replace channels position - YXC to CYX
        image = normalize_img_cyx(image)
        in_tensor = torch.from_numpy(image)

        # head tensors
        cached_obj = self.cache.try_get(idx)
        if cached_obj:
            return {"input": in_tensor, "target": cached_obj}

        target_tensors = self.codec.encode(sample_desc)
        self.cache.add(idx, target_tensors)
        return {"input": in_tensor, "target": target_tensors}

    def show_sample(self, idx):
        sample = self.__make_sample(idx)

        in_tensor = sample["input"]
        tgt_tensor = sample["target"]

        in_tensor = in_tensor.reshape(1, in_tensor.shape[0], in_tensor.shape[1], in_tensor.shape[2])
        tgt_tensor = add_batch_dim(tgt_tensor)

        imgs = visualize_inference(in_tensor, tgt_tensor, self.codec, self.index.get_id2class(), prediction=False)
        show_image_cv(imgs[0])
        return

    @staticmethod
    def set_noobj_prob(tns, pr_cnt, cls_cnt):
        for y_coord in range(tns.shape[0]):
            for x_coord in range(tns.shape[1]):
                for anchor_id in range(pr_cnt):
                    tns[y_coord, x_coord, anchor_id * cls_cnt + cls_cnt - 1] = 1
        return

    def get_codec(self):
        return self.codec


if __name__ == "__main__":
    dset = VocDetectionSSD(root="/home/igor/datasets/VOC_2007/trainval", logger=print)
    for i in range(100):
        dset.show_sample(i)

