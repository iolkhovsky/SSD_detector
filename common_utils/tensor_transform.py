import numpy as np
import torch
import unittest


TORCHVISION_NORM_MEAN = [0.485, 0.456, 0.406]
TORCHVISION_NORM_STD = [0.229, 0.224, 0.225]


def array_yxc2cyx(arr):
    arr = np.swapaxes(arr, 1, 2)
    arr = np.swapaxes(arr, 0, 1)
    return arr


def array_cyx2yxc(arr):
    arr = np.swapaxes(arr, 0, 1)
    arr = np.swapaxes(arr, 1, 2)
    return arr


def tensor_yxc2cyx(tns):
    return tns.permute(2, 0, 1)


def tensor_cyx2yxc(tns):
    return tns.permute(1, 2, 0)


def normalize_img_cyx(img):
    img = img.astype(dtype=np.float32)
    img = np.divide(img, 255.)  # 0...255 -> 0...1.
    r, g, b = img[0], img[1], img[2]
    r = np.divide(r - TORCHVISION_NORM_MEAN[0], TORCHVISION_NORM_STD[0])
    g = np.divide(g - TORCHVISION_NORM_MEAN[1], TORCHVISION_NORM_STD[1])
    b = np.divide(b - TORCHVISION_NORM_MEAN[2], TORCHVISION_NORM_STD[2])
    img[0] = r
    img[1] = g
    img[2] = b
    return img


def denormalize_img_cyx(img):
    r, g, b = img[0], img[1], img[2]
    r = np.multiply(r, TORCHVISION_NORM_STD[0]) + TORCHVISION_NORM_MEAN[0]
    g = np.multiply(g, TORCHVISION_NORM_STD[1]) + TORCHVISION_NORM_MEAN[1]
    b = np.multiply(b, TORCHVISION_NORM_STD[2]) + TORCHVISION_NORM_MEAN[2]
    img[0] = r
    img[1] = g
    img[2] = b
    img = np.multiply(img, 255.)  # 0...1. -> 0...255.
    img = img.astype(np.uint8)
    return img


def generate_random_tensor(*args):
    data = np.random.rand(*args).astype(np.float32)
    return torch.from_numpy(data)


def transfer_tuple_of_tensors(target_batch, device):
    target_batch = list(target_batch)
    for i in range(len(target_batch)):
        target_batch[i] = target_batch[i].to(device)
    return tuple(target_batch)


def add_batch_dim(tup_of_tensors):
    tup_of_tensors = list(tup_of_tensors)
    for i in range(len(tup_of_tensors)):
        tup_of_tensors[i] = tup_of_tensors[i].reshape(1, tup_of_tensors[i].shape[0], tup_of_tensors[i].shape[1],
                                                      tup_of_tensors[i].shape[2])
    return tuple(tup_of_tensors)


def tensor_bcyx2byxc(tns):
    tns = tns.permute(0, 2, 3, 1)
    return tns


def tensor_byxc2bcyx(tns):
    tns = tns.permute(0, 3, 1, 2)
    return tns


def custom_softmax(data_vector):
    return np.exp(data_vector) / np.sum(np.exp(data_vector))


class TestTransformFunctions(unittest.TestCase):

    def test_softmax(self):
        test_data = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
        target = [0.0236405, 0.0642617, 0.174681, 0.474833, 0.0236405, 0.0642617, 0.174681]
        for c, t in zip(list(custom_softmax(test_data)), list(np.asarray(target))):
            self.assertAlmostEqual(c, t, delta=1e-4)


if __name__ == "__main__":
    unittest.main()
