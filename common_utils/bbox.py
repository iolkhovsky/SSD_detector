import unittest


class BBox:

    def __init__(self, coords=None):
        self.x, self.y = None, None
        self.w, self.h = None, None
        if coords:
            self.set(coords)
        pass

    def set(self, coords):
        assert (len(list(coords)) == 4)
        self.x, self.y = coords[0], coords[1]
        self.w, self.h = coords[2], coords[3]
        return

    def set_norm(self, norm, imgsz):
        assert (len(list(imgsz)) == 2)
        assert (len(list(norm)) == 4)
        self.x = norm[0] * imgsz[0]
        self.y = norm[1] * imgsz[1]
        self.w = norm[2] * imgsz[0]
        self.h = norm[3] * imgsz[1]
        return

    def get_norm(self, imgsz):
        assert (len(list(imgsz)) == 2)
        norm_x = self.x / imgsz[0]
        norm_y = self.y / imgsz[1]
        norm_w = self.w / imgsz[0]
        norm_h = self.h / imgsz[1]
        return norm_x, norm_y, norm_w, norm_h

    def get_iou(self, other):
        return BBox.iou((self.x, self.y, self.w, self.h), (other.x, other.y, other.w, other.h))

    def make_valid(self, imgsz):
        assert len(list(imgsz)) == 2
        val_x = max(0, min(self.x, imgsz[0] - 1))
        val_y = max(0, min(self.y, imgsz[1] - 1))
        diff_x = val_x - self.x
        diff_y = val_y - self.y
        self.w -= diff_x
        self.h -= diff_y
        self.x = val_x
        self.y = val_y
        self.w = max(0, min(self.w, imgsz[0] - 1))
        self.h = max(0, min(self.h, imgsz[1] - 1))
        return


    @staticmethod
    def iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        intersection_x0 = max(x1, x2)
        intersection_x1 = min(x1+w1, x2+w2)
        if intersection_x1 <= intersection_x0:
            return 0.
        intersection_y0 = max(y1, y2)
        intersection_y1 = min(y1+h1, y2+h2)
        if intersection_y1 <= intersection_y0:
            return 0.
        intersection_area = (intersection_x1 - intersection_x0) * (intersection_y1 - intersection_y0)
        box1_area = w1 * h1
        box2_area = w2 * h2
        return intersection_area / (box1_area + box2_area - intersection_area)


class TestBBox(unittest.TestCase):

    def test_iou(self):
        box0 = BBox(coords=(0, 0, 100, 100))
        box1 = BBox(coords=(100, 100, 14, 15))
        self.assertAlmostEqual(box0.get_iou(box1), 0)
        box0 = BBox(coords=(0, 0, 100, 100))
        box1 = BBox(coords=(50, 50, 100, 100))
        self.assertAlmostEqual(box0.get_iou(box1), 0.142857143)


if __name__ == "__main__":
    unittest.main()
