import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from time import time

from common_utils.logger import Logger
from pascal_voc.voc_for_ssd import VocDetectionSSD
from pascal_voc.utils import make_dataloaders
from ssd.ssd_model import SSD
from ssd.ssd_training import ssd_loss_focal
from ssd.utils import save_model, load_model
from ssd.lr_schedule import load_training_schedule, LRScheduler, OptimizerWithSchedule
from common_utils.logger import LogDuration
from common_utils.parsing import parse_period_string
from common_utils.debug import SysArgsDebug
from ssd.ssd_inference import visualize_prediction_target, decode_prediction
from metrics.detection_metrics import mean_average_precision
from common_utils.tensor_transform import transfer_tuple_of_tensors


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_backbone", help="Initialize model's feature extractor with default weights")
    parser.add_argument("--checkpoint", help="Continue training from the checkpoint (abs path)")
    parser.add_argument("--dataset_path", help="Dataset (abs path)")
    parser.add_argument("--epochs_limit", help="Limit for epochs count per session")
    parser.add_argument("--batch_limit", help="Limit for batch count per epoch")
    parser.add_argument("--train_batch_size", help="Size of training batch")
    parser.add_argument("--val_batch_size", help="Size of validation batch")
    parser.add_argument("--autosave_period", help="Period set as string '10e'= 10 epochs, '10b'= 10 batches")
    parser.add_argument("--valid_period", help="Period set as string '10e'= 10 epochs, '10b'= 10 batches")
    parser.add_argument("--log_path", help="Path for storing log file")
    parser.add_argument("--use_gpu", help="Enable CUDA using")
    parser.add_argument("--lr", help="Default learning rate value (if not defined scheduler)")
    parser.add_argument("--lr_shedule_path", help="Path to configuration for LR scheduler")
    parser.add_argument("--dset_size_limit", help="Limit for dataset size (in samples)")
    parser.add_argument("--train_backbone", help="Enable training backbone")
    parser.add_argument("--dset_cache", help="Percentage of dataset keeped in cache")
    args = parser.parse_args()
    return args


def main():
    # cmd_args = parse_cmd_args()
    cmd_args = SysArgsDebug()
    train_logger = Logger(cmd_args.log_path)

    train_logger("********************************")
    train_logger("Start training session...", caller="training script")

    train_batch_size = int(cmd_args.train_batch_size)
    train_logger("Training batch size:", train_batch_size, caller="training script")
    val_batch_size = int(cmd_args.val_batch_size)
    train_logger("Validation batch size:", val_batch_size, caller="training script")
    epochs_limit = int(cmd_args.epochs_limit)
    train_logger("Epochs limit:", epochs_limit, caller="training script")
    use_cuda = torch.cuda.is_available() and int(cmd_args.use_gpu)
    train_logger("Use GPU:", use_cuda, caller="training script")
    autosave_period_epochs, autosave_period_batches = parse_period_string(cmd_args.autosave_period)
    train_logger("Autosave peiod(epochs/batches):", autosave_period_epochs, "/", autosave_period_batches,
                 caller="training script")
    validation_period_epochs, validation_period_batches = parse_period_string(cmd_args.valid_period)
    train_logger("Validation peiod(epochs/batches):", validation_period_epochs, "/", validation_period_batches,
                 caller="training script")

    dataset_volume_limit = None if cmd_args.dset_size_limit is None else int(cmd_args.dset_size_limit)
    trainval_dataset = VocDetectionSSD(root=cmd_args.dataset_path, logger=train_logger, cache=int(cmd_args.dset_cache),
                                       sz_limit=dataset_volume_limit)
    train_logger("Loaded trainval dataset: ", trainval_dataset, caller="training script")
    train_dataloader, val_dataloader = make_dataloaders(trainval_dataset,
                                                        train_batch_size=train_batch_size,
                                                        val_batch_size=val_batch_size)
    train_logger("Dataset cache: ", cmd_args.dset_cache, "%", caller="training script")

    pretrained_fext = int(cmd_args.pretrained_backbone)
    train_logger("Use pretrained backbone (feature extractor net): ", pretrained_fext, caller="training script")
    train_fext = int(cmd_args.train_backbone)
    train_logger("Enable training backbone: ", train_fext, caller="training script")
    model = load_model(cmd_args.checkpoint) if cmd_args.checkpoint else \
        SSD(pretrained=pretrained_fext, requires_grad=bool(train_fext))
    train_logger("Loaded detector model: ", model, caller="training script")

    # optimizer = torch.optim.SGD(model.parameters(), lr=float(cmd_args.lr), momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters())
    train_logger("Loaded optimizer: ", optimizer, caller="training script")

    batches_per_epochs = len(trainval_dataset) / train_batch_size
    train_logger("Batches per epoch: ", optimizer, caller="training script")

    lr_schedule_description = load_training_schedule(batches_per_epochs, path=cmd_args.lr_schedule_path)
    train_logger("LR schedule loaded from the ini-file: ", cmd_args.lr_schedule_path, caller="training script")
    lr_scheduler = LRScheduler(lr_schedule_description)
    train_logger("LR scheduler: ", lr_scheduler, caller="training script")
    scheduled_optimizer = OptimizerWithSchedule(optimizer, lr_scheduler)

    tboard_writer = SummaryWriter()

    torch.autograd.set_detect_anomaly(True)

    if use_cuda:
        model.cuda()

    global_step = 0
    summary_dict = {"LossValTotal": 0, "LossValClfP": 0, "LossValClfN": 0, "LossValClf": 0, "LossValLoc": 0}

    prev_stamp = time()

    for e in range(epochs_limit):
        for batch_idx, batch in enumerate(train_dataloader):

            summary_dict["Iteration duration"] = time() - prev_stamp
            prev_stamp = time()

            batch_processing_time = LogDuration()
            scheduled_optimizer.zero_grad()

            image_batch = batch["input"]
            target_batch = batch["target"]

            if use_cuda:
                image_batch = image_batch.to(device="cuda")
                # target_batch = target_batch.to(device="cuda")
                target_batch = transfer_tuple_of_tensors(target_batch, device="cuda")

            model.train()

            prediction_batch = model.forward(image_batch)
            loss_train_clf_p, loss_train_clf_n, train_loc_loss = ssd_loss_focal(target_batch, prediction_batch)
            loss_train_total = loss_train_clf_p + loss_train_clf_n + train_loc_loss
            loss_train_total.backward()
            scheduled_optimizer.step()

            summary_dict["Processing duration"] = batch_processing_time.get()
            summary_dict["Epoch"] = e
            summary_dict["Batch"] = batch_idx
            summary_dict["LossTrainTotal"] = loss_train_total.item()
            summary_dict["LossTrainClfP"] = loss_train_clf_p.item()
            summary_dict["LossTrainClfN"] = loss_train_clf_n.item()
            summary_dict["LossTrainClf"] = loss_train_clf_p.item() + loss_train_clf_n.item()
            summary_dict["LossTrainLoc"] = train_loc_loss.item()
            summary_dict["Global step"] = global_step
            summary_dict["Learning rate"] = scheduled_optimizer.scheduler.get_pars()[0]
            summary_dict["Batch size"] = train_batch_size

            if autosave_period_batches and global_step:
                if global_step % autosave_period_batches == 0:
                    model = save_model(model, "Ep" + str(e) + "Btch" + str(batch_idx))
                    train_logger("Model saved at epoch: " + str(e) + " iteration: " + str(batch_idx))

            if validation_period_batches and global_step:
                if global_step % validation_period_batches == 0:
                    model.eval()

                    for sample in val_dataloader:
                        image_batch = sample["input"]
                        target_batch = sample["target"]
                        if use_cuda:
                            image_batch = image_batch.to(device="cuda")
                            # target_batch = target_batch.to(device="cuda")
                            target_batch = transfer_tuple_of_tensors(target_batch, device="cuda")
                        prediction_batch = model.forward(image_batch)
                        if use_cuda:
                            # prediction_batch = prediction_batch.cpu()
                            prediction_batch = transfer_tuple_of_tensors(prediction_batch, device="cpu")
                            # target_batch = target_batch.cpu()
                            target_batch = transfer_tuple_of_tensors(target_batch, device="cpu")
                            image_batch = image_batch.cpu()

                        loss_val_clf_p, loss_val_clf_n, val_loc_loss = ssd_loss_focal(target_batch, prediction_batch)
                        val_total_loss = loss_val_clf_p + loss_val_clf_n + val_loc_loss

                        summary_dict["LossValTotal"] = val_total_loss.item()
                        summary_dict["LossValClfP"] = loss_val_clf_p.item()
                        summary_dict["LossValClfN"] = loss_val_clf_n.item()
                        summary_dict["LossValClf"] = loss_val_clf_p.item() + loss_val_clf_n.item()
                        summary_dict["LossValLoc"] = val_loc_loss.item()

                        codec = trainval_dataset.codec
                        id2class = trainval_dataset.index.get_id2class()

                        pred_imgs, target_imgs = visualize_prediction_target(image_batch, prediction_batch, target_batch,
                                                                             codec, id2class, to_tensors=True)

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
                                summary_dict["Valid AP "+str(label)] = value
                        break

                    model.train()

            if autosave_period_epochs and e:
                if (e % autosave_period_epochs == 0) and batch_idx == 0:
                    model = save_model(model, "Ep" + str(e) + "Btch" + str(batch_idx))
                    train_logger("Model saved at epoch: " + str(e) + " iteration: " + str(batch_idx))

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

            train_logger("****************Iteration summary****************", caller="training script")
            train_logger.log_dict(summary_dict, caller="training script")
            global_step += 1
    return


if __name__ == "__main__":
    main()
