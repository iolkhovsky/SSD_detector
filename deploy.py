import cv2
import argparse

from common_utils.tensor_transform import *
from ssd.utils import load_model
from ssd.ssd_inference import visualize_inference
from ssd.ssd_codec import SSDCodec, SSDBoxCodec
from ssd.constraints import DEFAULT_FMAP_SIZES, DEFAULT_TARGET_SIZE
from pascal_voc.voc_index import DEFAULT_LABEL_MAP
from pascal_voc.voc_index import VocIndex


def make_tensor_for_net(cv_img, in_img_sz=(300, 300), to_gpu=False):
    image = cv2.resize(cv_img, in_img_sz)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # reverse bgr to rgb
    image = array_yxc2cyx(image)  # replace channels position - YXC to CYX
    image = normalize_img_cyx(image)
    in_tensor = torch.from_numpy(image).reshape(1, 3, in_img_sz[0], in_img_sz[1])  # batch-line tensor
    if to_gpu:
        return in_tensor.cuda()
    else:
        return in_tensor


model_path = "Ep74Btch126_SSD_Mobilenetv2_6fm6p21c_2020_02_14_06_45_00.torchmodel"

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="Path to model's checkpoint to load")
parser.add_argument("--threshold", help="Threshold for visualization")
args = parser.parse_args()

if args.model:
    model_path = args.model
print("Loaded model from: ", model_path)

use_cuda = torch.cuda.is_available()

model = load_model(model_path, logger=print)
model.eval()
if use_cuda:
    model = model.cuda()
box_codec = SSDBoxCodec(DEFAULT_TARGET_SIZE, DEFAULT_FMAP_SIZES)
codec = SSDCodec(box_codec=box_codec)
class2id, id2class = VocIndex.load_label_map(DEFAULT_LABEL_MAP)

threshold = 1.0
if args.threshold:
    threshold = float(args.threshold)
print("Threshold for detection is set as: ", threshold)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    src_frame_sz = frame.shape
    img_tensor = make_tensor_for_net(frame, to_gpu=use_cuda)
    prediction = model.forward(img_tensor)
    if use_cuda:
        img_tensor = img_tensor.cpu()
        prediction = transfer_tuple_of_tensors(prediction, device="cpu")

    imgs = visualize_inference(img_tensor, prediction, codec, id2class, prediction=True, prob_thr=threshold)

    out_image = cv2.resize(imgs[0], (src_frame_sz[1], src_frame_sz[0]))
    cv2.imshow("Stream", out_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



