import matplotlib.pyplot as plt
import numpy as np
import cv2


DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX
DEFAULT_FONT_SCALE = 0.3


def show_image_matplot(im):
    if type(im) != np.ndarray:
        raise RuntimeError("Not an ndarray put into show_image util")
    fig, ax = plt.subplots(1)
    ax.imshow(im)
    plt.show()
    return


def show_image_cv(im):
    if type(im) != np.ndarray:
        raise RuntimeError("Not an ndarray put into show_image util")
    cv2.imshow("Image", im)
    cv2.waitKey(1)
    return


def draw_object_mark(image, class_label, class_box, prob=0.5):
    class_box.x = int(class_box.x)
    class_box.y = int(class_box.y)
    class_box.w = int(class_box.w)
    class_box.h = int(class_box.h)
    if (prob is None) or np.isnan(prob):
        prob = 0.0
    box_thickness = int((prob - 0.5) * 16)  # max width = 8 with prob = 1.0
    if box_thickness < 1:
        box_thickness = 1
    pt1 = min(max(class_box.x, 0), image.shape[1]), min(max(class_box.y, 0), image.shape[0])
    pt2 = min(max(class_box.x + class_box.w - 1, 0), image.shape[1]), \
          min(max(class_box.y + class_box.h - 1, 0), image.shape[0])
    image = cv2.rectangle(img=image,
                          pt1=pt1,
                          pt2=pt2,
                          color=(0, 0, 255),
                          thickness=box_thickness)
    org_x, org_y = class_box.x - 7, class_box.y - 7
    origin = min(max(org_x, 0), image.shape[1]), min(max(org_y, 0), image.shape[0])
    image = cv2.putText(image, class_label, origin, DEFAULT_FONT, DEFAULT_FONT_SCALE, (0, 0, 255),
                        1, cv2.LINE_AA)
    return image



