from os.path import isfile
from os.path import basename, splitext
from ssd.lr_schedule import LRBehavior
import yaml


class PipelineConfig:

    def __init__(self, args):
        self.pipeline_config = args.pipeline_config
        self.pretrained_backbone = args.pretrained_backbone
        self.checkpoint = args.checkpoint
        self.dataset_path = args.dataset_path
        self.epochs_limit = args.epochs_limit
        self.train_batch_size = args.train_batch_size
        self.val_batch_size = args.val_batch_size
        self.autosave_period = args.autosave_period
        self.valid_period = args.valid_period
        self.log_path = args.log_path
        self.use_gpu = args.use_gpu
        self.lr = args.lr
        self.dset_size_limit = args.dset_size_limit
        self.train_backbone = args.train_backbone
        self.dset_cache = args.dset_cache

        self.lr_schedule = [{"value_start": 1e-4, "value_stop": 1e-4, "type": LRBehavior.LR_CONSTANT, "duration": 1}]


def parse_pipeline(pipeline_path, default=None):
    yml_dict = {}
    with open(pipeline_path, "r") as f:
        yml_dict = yaml.load(f, Loader=yaml.SafeLoader)
    if "model" in yml_dict.keys():
        model_config = yml_dict["model"]
        if "checkpoint" in model_config.keys():
            default.checkpoint = model_config["checkpoint"]
        if "pretrained_backbone" in model_config.keys():
            default.checkpoint = int(model_config["pretrained_backbone"])
        if "train_backbone" in model_config.keys():
            default.checkpoint = int(model_config["train_backbone"])
    if "dataset" in yml_dict.keys():
        dataset_config = yml_dict["dataset"]
        if "path" in dataset_config.keys():
            default.dataset_path = dataset_config["path"]
        if "size_limit" in dataset_config.keys():
            default.dset_size_limit = int(dataset_config["size_limit"])
        if "cache" in dataset_config.keys():
            default.dset_cache = int(dataset_config["cache"])
    if "training" in yml_dict.keys():
        training_config = yml_dict["training"]
        if "epoch_limit" in training_config.keys():
            default.epochs_limit = int(training_config["epoch_limit"])
        if "batch_size" in training_config.keys():
            default.train_batch_size = int(training_config["batch_size"])
        if "learning_rate" in training_config.keys():
            default.lr = float(training_config["learning_rate"])
        if "optimizer" in training_config.keys():
            default.optimizer = training_config["optimizer"]
        if "scheduler" in training_config.keys():
            default.use_scheduler = int(training_config["scheduler"])
        if "lr_lambda" in training_config.keys():
            default.lr_lambda = float(training_config["lr_lambda"])
        if "scheduler_period" in training_config.keys():
            default.lr_step_period = int(training_config["scheduler_period"])
    if "validation" in yml_dict.keys():
        validation_config = yml_dict["validation"]
        if "batch_size" in validation_config.keys():
            default.val_batch_size = int(validation_config["batch_size"])
        if "period" in validation_config.keys() and "unit" in validation_config.keys():
            suffix = "b" if validation_config["unit"] == "batch" else "e"
            default.valid_period = str(validation_config["period"]) + suffix
        if "visualiz_conf_threshold" in validation_config.keys():
            default.visual_conf_thresh = float(validation_config["visualiz_conf_threshold"])
    if "autosave" in yml_dict.keys():
        autosave_config = yml_dict["autosave"]
        if "period" in autosave_config.keys() and "unit" in autosave_config.keys():
            suffix = "b" if autosave_config["unit"] == "batch" else "e"
            default.autosave_period = str(autosave_config["period"]) + suffix
    if "gpu" in yml_dict.keys():
        gpu_settings = yml_dict["gpu"]
        if "enable" in gpu_settings.keys():
            default.use_gpu = int(gpu_settings["enable"])
    return default


def configure_pipeline(cmdline_args):
    config = PipelineConfig(cmdline_args)
    if isfile(cmdline_args.pipeline_config) and splitext(basename(cmdline_args.pipeline_config))[1] == ".yml":
        config = parse_pipeline(cmdline_args.pipeline_config, default=config)
    return config
