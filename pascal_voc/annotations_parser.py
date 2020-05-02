from pascal_voc.descriptors import *
import xml.etree.ElementTree as elemtree


def get_sample_descriptor_from_xml(dataset_root, annotation, class2id):
    out = SampleDescriptor()
    tree = elemtree.parse(annotation)
    root = tree.getroot()
    unique_labels = set()

    # first find filename and image size
    for child in root:
        if child.tag == "filename":
            out.abs_path = dataset_root + "/JPEGImages/" + child.text
        if child.tag == "size":
            w = None
            h = None
            c = None
            for _child in child:
                if _child.tag == "width":
                    w = int(_child.text)
                elif _child.tag == "height":
                    h = int(_child.text)
                elif _child.tag == "depth":
                    c = int(_child.text)
            out.img_size = (c, h, w)

    # then find bboxes
    for child in root:
        if child.tag == "object":
            objdesc = ObjectDescriptor()
            for _child in child:
                if _child.tag == "name":
                    objdesc.class_label = _child.text
                    unique_labels.add(objdesc.class_label)
                if _child.tag == "bndbox":
                    x0, x1, y0, y1 = None, None, None, None
                    for _par in _child:
                        if _par.tag == "xmin":
                            x0 = int(float(_par.text))
                        elif _par.tag == "ymin":
                            y0 = int(float(_par.text))
                        elif _par.tag == "xmax":
                            x1 = int(float(_par.text))
                        elif _par.tag == "ymax":
                            y1 = int(float(_par.text))
                    objdesc.x = x0
                    objdesc.y = y0
                    objdesc.w = x1 - x0 + 1
                    objdesc.h = y1 - y0 + 1

            objdesc.class_id = class2id[objdesc.class_label]
            out.objects.append(objdesc)
    return out, unique_labels
