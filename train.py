import argparse
import multiprocessing
import os
from os.path import isfile
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
import torchvision
from time import time
import sys

from common_utils.logger import Logger
from common_utils.pipeline_configuration import configure_pipeline
from pascal_voc.voc_for_ssd import VocDetectionSSD
from pascal_voc.utils import make_dataloaders
from ssd.ssd_model import SSD
from ssd.ssd_training import ssd_loss_focal
from ssd.utils import save_model, load_model
from common_utils.logger import LogDuration
from common_utils.parsing import parse_period_string
from ssd.ssd_inference import visualize_prediction_target, decode_prediction
from metrics.detection_metrics import mean_average_precision
from common_utils.tensor_transform import transfer_tuple_of_tensors


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_config", type=str, default="training_pipeline.yml")
    parser.add_argument("--pretrained_backbone", type=int, default=1,
                        help="Initialize model's feature extractor with default weights")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Continue training from the checkpoint (abs path)")
    parser.add_argument("--dataset_path", type=str, default="/home/igor/datasets/VOC_2007/trainval",
                        help="Dataset (abs path)")
    parser.add_argument("--epochs_limit", type=int, default=40,
                        help="Limit for epochs count per session")
    parser.add_argument("--train_batch_size", type=int, default=16,
                        help="Size of training batch")
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help="Size of validation batch")
    parser.add_argument("--autosave_period", type=str, default="20b",
                        help="Period set as string '10e'= 10 epochs, '10b'= 10 batches")
    parser.add_argument("--valid_period", type=str, default="1b",
                        help="Period set as string '10e'= 10 epochs, '10b'= 10 batches")
    parser.add_argument("--log_path", type=str, default="training.log",
                        help="Path for storing log file")
    parser.add_argument("--use_gpu", type=int, default=0,
                        help="Enable CUDA using")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Default learning rate value (if not defined scheduler)")
    parser.add_argument("--dset_size_limit", type=int, default=320,
                        help="Limit for dataset size (in samples)")
    parser.add_argument("--train_backbone", type=int, default=1,
                        help="Enable training backbone")
    parser.add_argument("--dset_cache", type=int, default=100,
                        help="Percentage of dataset keeped in cache")
    parser.add_argument("--visual_conf_thresh", type=float, default=0.5)
    parser.add_argument("--use_scheduler", type=int, default=1)
    parser.add_argument("--lr_lambda", type=float, default=0.95)
    parser.add_argument("--lr_step_period", type=int, default=1)
    args = parser.parse_args()
    return args


def model_autosave(model, epoch, batch, global_step, period_epochs, period_batches, logger=print):
    batch_period_flag = period_batches and ((global_step + 1) % period_batches == 0)
    epoch_period_flag = period_epochs and (((epoch + 1) % period_epochs == 0) and batch == 0)
    if batch_period_flag or epoch_period_flag:
        save_model(model, "Ep" + str(epoch) + "Btch" + str(batch))
        logger("Model saved at epoch: " + str(epoch) + " iteration: " + str(batch))


def train_ssd(model, train_dataloader, val_dataloader, validation_period_batches, optimizer, scheduler, epochs_cnt,
              tboard_writer, id2class, codec, autosave_period_batches=None, autosave_period_epochs=None, logger=print,
              use_cuda=False, visual_conf_threshold=0.5):
    global_step = 0
    summary_dict = {"LossValTotal": 0, "LossValClfP": 0, "LossValClfN": 0, "LossValClf": 0, "LossValLoc": 0}
    prev_stamp = time()

    for e in range(epochs_cnt):
        for batch_idx, batch in enumerate(train_dataloader):

            summary_dict["Iteration duration"] = time() - prev_stamp
            prev_stamp = time()

            batch_processing_time = LogDuration()
            optimizer.zero_grad()

            image_batch = batch["input"]
            target_batch = batch["target"]

            if use_cuda:
                image_batch = image_batch.to(device="cuda")
                target_batch = transfer_tuple_of_tensors(target_batch, device="cuda")

            model.train()

            prediction_batch = model.forward(image_batch)
            loss_train_clf_p, loss_train_clf_n, train_loc_loss = ssd_loss_focal(target_batch, prediction_batch)
            loss_train_total = loss_train_clf_p + loss_train_clf_n + train_loc_loss
            loss_train_total.backward()
            optimizer.step()

            summary_dict["Processing duration"] = batch_processing_time.get()
            summary_dict["Epoch"] = e
            summary_dict["Batch"] = batch_idx
            summary_dict["LossTrainTotal"] = loss_train_total.item()
            summary_dict["LossTrainClfP"] = loss_train_clf_p.item()
            summary_dict["LossTrainClfN"] = loss_train_clf_n.item()
            summary_dict["LossTrainClf"] = loss_train_clf_p.item() + loss_train_clf_n.item()
            summary_dict["LossTrainLoc"] = train_loc_loss.item()
            summary_dict["Global step"] = global_step
            summary_dict["Learning rate"] = scheduler.get_last_lr()[0]
            summary_dict["Batch size"] = len(image_batch)

            model_autosave(model=model, epoch=e, batch=batch_idx, global_step=global_step,
                           period_epochs=autosave_period_epochs, period_batches=autosave_period_batches,
                           logger=logger)
            if use_cuda:
                model = model.cuda()

            if validation_period_batches:
                if (global_step + 1) % validation_period_batches == 0:

                    model.eval()
                    sample = next(iter(val_dataloader))

                    image_batch = sample["input"]
                    target_batch = sample["target"]
                    if use_cuda:
                        image_batch = image_batch.cuda()
                        target_batch = transfer_tuple_of_tensors(target_batch, device="cuda")
                    prediction_batch = model.forward(image_batch)
                    if use_cuda:
                        prediction_batch = transfer_tuple_of_tensors(prediction_batch, device="cpu")
                        target_batch = transfer_tuple_of_tensors(target_batch, device="cpu")
                        image_batch = image_batch.cpu()

                    loss_val_clf_p, loss_val_clf_n, val_loc_loss = ssd_loss_focal(target_batch, prediction_batch)
                    val_total_loss = loss_val_clf_p + loss_val_clf_n + val_loc_loss

                    summary_dict["LossValTotal"] = val_total_loss.item()
                    summary_dict["LossValClfP"] = loss_val_clf_p.item()
                    summary_dict["LossValClfN"] = loss_val_clf_n.item()
                    summary_dict["LossValClf"] = loss_val_clf_p.item() + loss_val_clf_n.item()
                    summary_dict["LossValLoc"] = val_loc_loss.item()

                    pred_imgs, target_imgs = visualize_prediction_target(image_batch, prediction_batch,
                                                                         target_batch,
                                                                         codec, id2class, to_tensors=True,
                                                                         threshold=visual_conf_threshold)

                    img_grid_pred = torchvision.utils.make_grid(pred_imgs)
                    img_grid_tgt = torchvision.utils.make_grid(target_imgs)
                    tboard_writer.add_image('Valid/Predicted', img_tensor=img_grid_pred,
                                            global_step=global_step, dataformats='CHW')
                    tboard_writer.add_image('Valid/Target', img_tensor=img_grid_tgt,
                                            global_step=global_step, dataformats='CHW')

                    predictions = decode_prediction(prediction_batch, codec, id2class, prediction=True,
                                                    threshold=0.0)
                    targets = decode_prediction(target_batch, codec, id2class, prediction=False)
                    quality = mean_average_precision(predictions, targets)
                    for label, value in quality.items():
                        if label == "map":
                            tboard_writer.add_scalar("AP/mAP", value, global_step)
                            summary_dict["Valid mAP"] = value
                        else:
                            tboard_writer.add_scalar("AP/" + label, value, global_step)
                            summary_dict["Valid AP " + str(label)] = value

                    model.train()

            tboard_writer.add_scalar('Loss/TrainTotal', summary_dict["LossTrainTotal"], global_step)
            tboard_writer.add_scalar('Loss/TrainClassP', summary_dict["LossTrainClfP"], global_step)
            tboard_writer.add_scalar('Loss/TrainClassN', summary_dict["LossTrainClfN"], global_step)
            tboard_writer.add_scalar('Loss/TrainClass', summary_dict["LossTrainClf"], global_step)
            tboard_writer.add_scalar('Loss/TrainLocal', summary_dict["LossTrainLoc"], global_step)

            tboard_writer.add_scalar('Loss/ValTotal', summary_dict["LossValTotal"], global_step)
            tboard_writer.add_scalar('Loss/ValClassP', summary_dict["LossValClfP"], global_step)
            tboard_writer.add_scalar('Loss/ValClassN', summary_dict["LossValClfN"], global_step)
            tboard_writer.add_scalar('Loss/ValClass', summary_dict["LossValClf"], global_step)
            tboard_writer.add_scalar('Loss/ValLocal', summary_dict["LossValLoc"], global_step)

            tboard_writer.add_scalar('Train/LearningRate', summary_dict["Learning rate"], global_step)
            tboard_writer.add_scalar('Train/IterationTime', summary_dict["Iteration duration"], global_step)

            logger("****************Iteration summary****************", caller="training script")
            logger.log_dict(summary_dict, caller="training script")
            global_step += 1
        if scheduler is not None:
            scheduler.step()
        logger.log_dict(f"Epoch #{e} completed", caller="training script")


def main():
    args = configure_pipeline(parse_cmd_args())
    train_logger = Logger(args.log_path)

    train_logger("********************************")
    train_logger("Start training session...", caller="training script")
    train_logger("Training batch size:", args.train_batch_size, caller="training script")
    train_logger("Validation batch size:", args.val_batch_size, caller="training script")
    train_logger("Epochs limit:", args.epochs_limit, caller="training script")
    use_cuda = torch.cuda.is_available() and args.use_gpu
    train_logger("Use GPU:", use_cuda, caller="training script")
    autosave_period_epochs, autosave_period_batches = parse_period_string(args.autosave_period)
    train_logger("Autosave peiod(epochs/batches):", autosave_period_epochs, "/", autosave_period_batches,
                 caller="training script")
    validation_period_epochs, validation_period_batches = parse_period_string(args.valid_period)
    train_logger("Validation peiod(epochs/batches):", validation_period_epochs, "/", validation_period_batches,
                 caller="training script")

    trainval_dataset = VocDetectionSSD(root=args.dataset_path, logger=train_logger, cache=int(args.dset_cache),
                                       sz_limit=args.dset_size_limit)
    train_logger("Loaded trainval dataset: ", trainval_dataset, caller="training script")
    train_dataloader, val_dataloader = make_dataloaders(trainval_dataset,
                                                        train_batch_size=args.train_batch_size,
                                                        val_batch_size=args.val_batch_size)
    train_logger("Dataset cache: ", args.dset_cache, "%", caller="training script")

    train_logger("Use pretrained backbone (feature extractor net): ", args.pretrained_backbone,
                 caller="training script")
    train_logger("Enable training backbone: ", args.train_backbone, caller="training script")
    model = load_model(args.checkpoint) if isfile(args.checkpoint) else \
        SSD(pretrained=args.pretrained_backbone, requires_grad=bool(args.train_backbone))
    if use_cuda:
        model.cuda()
    else:
        available_cpu_cnt = multiprocessing.cpu_count()
        train_logger("Found cpu cores: ", available_cpu_cnt, caller="training script")
        torch_workers = available_cpu_cnt - 2 if available_cpu_cnt > 2 else available_cpu_cnt
        torch.set_num_threads(torch_workers)
        train_logger("Setting threads cnt: ", torch_workers, caller="training script")
    train_logger("Loaded detector model: ", model, caller="training script")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_logger("Loaded optimizer: ", optimizer, caller="training script")
    scheduler = None
    if args.use_scheduler:
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: args.lr_lambda ** (epoch // args.lr_step_period))
    torch.autograd.set_detect_anomaly(True)

    batches_per_epochs = len(trainval_dataset) / args.train_batch_size
    train_logger("Batches per epoch: ", batches_per_epochs, caller="training script")

    train_logger("Confidence threshold for visualization: ", args.visual_conf_thresh, caller="training script")

    tboard_writer = SummaryWriter()

    try:
        train_ssd(model=model,
                  train_dataloader=train_dataloader,
                  val_dataloader=val_dataloader,
                  validation_period_batches=validation_period_batches,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  epochs_cnt=args.epochs_limit,
                  tboard_writer=tboard_writer,
                  id2class=trainval_dataset.index.get_id2class(),
                  codec=trainval_dataset.codec,
                  autosave_period_batches=autosave_period_batches,
                  autosave_period_epochs=autosave_period_epochs,
                  logger=train_logger,
                  use_cuda=use_cuda,
                  visual_conf_threshold=args.visual_conf_thresh)
    except KeyboardInterrupt:
        save_model(model, "interrupted_train")
        train_logger("Training interrupted, model saved", caller="training script")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


if __name__ == "__main__":
    main()
