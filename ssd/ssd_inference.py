from common_utils.visualization import *
from common_utils.tensor_transform import *
from common_utils.bbox import *


def decode_prediction(prediction_batch, codec, id2class, prediction=False, threshold=0.5):
    # prediction batch is a tuple of tensors
    batch_sz = len(prediction_batch[0])
    out = []
    for i in range(batch_sz):
        pred_heads = (prediction_batch[0][i], prediction_batch[1][i],
                      prediction_batch[2][i], prediction_batch[3][i],
                      prediction_batch[4][i], prediction_batch[5][i],
                      prediction_batch[6][i], prediction_batch[7][i],
                      prediction_batch[8][i], prediction_batch[9][i],
                      prediction_batch[10][i], prediction_batch[11][i])
        detections = codec.decode(pred_heads, id2class, prediction=prediction, threshold=threshold)
        out.append(detections)
    return out


def decode_input_tensor(input_batch):
    out = []
    for image in input_batch:
        image = denormalize_img_cyx(image.numpy())
        image = array_cyx2yxc(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        out.append(image)
    return out


def visualize_prediction_target(input_batch, prediction_batch, target_batch, codec, id2class, to_tensors=False,
                                threshold=0.5):
    detections = decode_prediction(prediction_batch, codec, id2class, prediction=True, threshold=threshold)
    gtruth = decode_prediction(target_batch, codec, id2class, prediction=False, threshold=threshold)
    images = decode_input_tensor(input_batch)
    tgt_imgs = make_pred_images(images, gtruth, to_tensors)
    pred_imgs = make_pred_images(images, detections, to_tensors)
    return pred_imgs, tgt_imgs


def visualize_inference(input_batch, output_batch, codec, id2class, prediction=False, to_tensors=False, prob_thr=0.0):
    detections = decode_prediction(output_batch, codec, id2class, prediction=prediction, threshold=prob_thr)
    images = decode_input_tensor(input_batch)
    pred_imgs = make_pred_images(images, detections, to_tensors)
    return pred_imgs


def make_pred_images(imgs, dets, to_tensors):
    out = []
    for i in range(len(imgs)):
        img = imgs[i].copy()
        det = dets[i]
        for d in det:
            draw_object_mark(img, d.label, BBox((d.x, d.y, d.w, d.h)), prob=d.p)
        if to_tensors:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = array_yxc2cyx(img)
            img = torch.from_numpy(img)
        out.append(img)
    return out
