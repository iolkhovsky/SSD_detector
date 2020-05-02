

class SysArgsDebug:

    def __init__(self):
        self.pretrained_backbone = "1"
        self.checkpoint = None
        self.dataset_path = "/home/igor/datasets/VOC_2007/trainval"
        self.epochs_limit = "40"
        self.batch_limit = None
        self.train_batch_size = "32"
        self.val_batch_size = "8"
        self.autosave_period = "20b"
        self.valid_period = "2b"
        self.log_path = None
        self.use_gpu = "0"
        self.lr = "0.003"
        self.lr_schedule_path = "configs/lr_schedule.txt"
        self.dset_size_limit = "320"
        self.train_backbone = "0"
        self.dset_cache = "100"
        return
