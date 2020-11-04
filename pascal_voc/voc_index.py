from os.path import isdir, isfile
from common_utils.file_utils import get_files_list
import random
from pascal_voc.annotations_parser import get_sample_descriptor_from_xml
from pascal_voc.descriptors import *
from common_utils.visualization import show_image_cv
from common_utils.visualization_constraints import *
import cv2


DEFAULT_LABEL_MAP = "/home/igor/PycharmProjects/ssd_detector/configs/label_maps/label_map_voc.txt"


class VocIndex:

    def __init__(self, root_path=None, index_name=None, size_limit=None, label_map=DEFAULT_LABEL_MAP):
        self.__root = root_path
        self.__index_name = index_name
        self.__sz_limit = size_limit

        self.__sample_descriptions = list()
        self.__annotations = None
        self.__objects_labels = set()
        self.__label2id_map, self.__id2label_map = self.load_label_map(label_map)

        self.__current_idx = 0

        if root_path:
            self.load()
        return

    def __iter__(self):
        return self

    def __next__(self):
        if self.__sample_descriptions:
            if self.__current_idx >= len(self.__sample_descriptions):
                raise StopIteration
            else:
                return self.__sample_descriptions[self.__current_idx]
        else:
            raise RuntimeError("Index is empty")

    def __str__(self):
        text_description = "Pascal VOC index <" + str(self.__index_name) + ">"
        if self.__sample_descriptions:
            text_description += " Size: " + str(len(self.__sample_descriptions))
        else:
            text_description += " Not loaded"
        return text_description

    def load(self, root_path=None):
        if root_path:
            self.__root = root_path
        if not self.__root:
            raise RuntimeError("Index root hasnt defined for Index<" + str(self.__index_name) + ">")
        if self.__root is not None and isdir(self.__root):
            self.__annotations = get_files_list(path=self.__root + "/Annotations", filter_key=".xml")
            if self.__sz_limit:
                # shuffle annotations list
                random.shuffle(self.__annotations)
                # get clipped list
                self.__annotations = self.__annotations[:self.__sz_limit]
            self.__compile_descriptions()
        else:
            raise RuntimeError("Invalid root path <" + self.__root + "> for Index<" + str(self.__index_name) + ">")

    def get_sample(self, idx):
        if idx >= len(self.__sample_descriptions):
            raise StopIteration
        else:
            return self.__sample_descriptions[idx]

    def __compile_descriptions(self):
        for ann in self.__annotations:
            sample_desc, unique_labels = get_sample_descriptor_from_xml(self.__root, ann, self.__label2id_map)
            self.__sample_descriptions.append(sample_desc)
            self.__objects_labels = self.__objects_labels.union(unique_labels)
        return

    @staticmethod
    def load_label_map(label_map_path):
        class2id_dict = dict()
        id2class_dict = dict()
        if isfile(label_map_path):
            with open(label_map_path, "r") as mf:
                pair = mf.readline().replace("\n", "")
                while pair:
                    pair = pair.split()
                    class2id_dict[pair[1]] = int(pair[0])
                    id2class_dict[int(pair[0])] = pair[1]
                    pair = mf.readline()
        else:
            raise RuntimeError("Label map file path is invalid")
        return class2id_dict, id2class_dict

    def show_sample(self, idx):
        sample_desc = self.get_sample(idx)
        image = cv2.imread(sample_desc.abs_path)
        for obj in sample_desc.objects:
            color = BOX_CV_COLORS[self.__label2id_map[obj.class_label]]
            image = cv2.rectangle(img=image, pt1=(obj.x, obj.y), pt2=(obj.x + obj.w - 1, obj.y + obj.h - 1),
                                color=color, thickness=2)
            # label
            org_x, org_y = obj.x - 5, obj.y - 5
            if org_x < 0:
                org_x = 0
            if org_y < 0:
                org_y = 0
            image = cv2.putText(image, obj.class_label, (org_x, org_y), FONT, FONT_SCALE, color,
                              1, cv2.LINE_AA)
        show_image_cv(image)
        return

    def __getitem__(self, item):
        return self.get_sample(item)

    def get_class2id(self):
        return self.__label2id_map

    def get_id2class(self):
        return self.__id2label_map

    def __len__(self):
        return len(self.__sample_descriptions)


if __name__ == "__main__":
    index = VocIndex("/home/igor/datasets/VOC_2007/trainval", label_map=DEFAULT_LABEL_MAP)
    for idx in range(len(index)):
        if index[idx].abs_path[-7:] == "672.jpg":
            index.show_sample(idx)
            print(idx, index[idx].abs_path)
    print("ready")
