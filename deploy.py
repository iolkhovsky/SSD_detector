import cv2
import argparse

from common_utils.tensor_transform import *
from ssd.utils import load_model
from ssd.ssd_inference import visualize_inference
from ssd.ssd_codec import SSDCodec, SSDBoxCodec
from ssd.constraints import DEFAULT_FMAP_SIZES, DEFAULT_TARGET_SIZE
from pascal_voc.voc_index import DEFAULT_LABEL_MAP
from pascal_voc.voc_index import VocIndex


def parse_cmdline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ssd.torchmodel", help="Path to model's checkpoint to load")
    parser.add_argument("--cam", type=int, default=2, help="Webcamera id")
    parser.add_argument("--threshold", type=float, default=0.2, help="Threshold for visualization")
    return parser.parse_args()


def make_tensor_for_net(cv_img, in_img_sz=(300, 300), to_gpu=False):
    image = cv2.resize(cv_img, in_img_sz)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = array_yxc2cyx(image)
    image = normalize_img_cyx(image)
    in_tensor = torch.from_numpy(image).reshape(1, 3, in_img_sz[0], in_img_sz[1])  # batch-line tensor
    if to_gpu:
        return in_tensor.cuda()
    else:
        return in_tensor


if __name__ == "__main__":
    args = parse_cmdline()
    use_cuda = torch.cuda.is_available()
    model = load_model(args.model)
    print("Loaded model from: ", args.model)
    if use_cuda:
        model = model.cuda()
    model.eval()

    box_codec = SSDBoxCodec(DEFAULT_TARGET_SIZE, DEFAULT_FMAP_SIZES)
    codec = SSDCodec(box_codec=box_codec)
    class2id, id2class = VocIndex.load_label_map(DEFAULT_LABEL_MAP)

    cap = cv2.VideoCapture(args.cam)
    while True:
        ret, frame = cap.read()
        src_frame_sz = frame.shape
        img_tensor = make_tensor_for_net(frame, to_gpu=use_cuda)
        prediction = model.forward(img_tensor)
        if use_cuda:
            img_tensor = img_tensor.cpu()
            prediction = transfer_tuple_of_tensors(prediction, device="cpu")

        imgs = visualize_inference(img_tensor, prediction, codec, id2class, prediction=True, prob_thr=args.threshold,
                                   nms=True)

        out_image = cv2.resize(imgs[0], (src_frame_sz[1], src_frame_sz[0]))
        cv2.imshow("Stream", out_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



