import torch
from common_utils.timestamp import get_timestamp
from os.path import isfile


MODELS_REPOSITORY_PATH = "/home/igor/models_checkpoints/"
CUDA_IS_AVAILABLE = torch.cuda.is_available()


def compile_checkpoint_name(name, hint=None):
    name = name.replace(" ", "_")
    file_name = MODELS_REPOSITORY_PATH
    if hint is not None:
        file_name += hint+"_"
    file_name += name + "_"
    file_name += get_timestamp()
    file_name += ".torchmodel"
    return file_name


def load_model(path):
    return torch.load(path)


def save_model(torch_model, convert_to_cpu=True, hint=None, logger=None):
    fname = compile_checkpoint_name(str(torch_model), hint)
    if convert_to_cpu:
        torch_model = torch_model.cpu()
    torch.save(torch_model, fname)
    if logger:
        logger(f"Model '{fname}' successfully saved.")
    return torch_model


def get_scale_for_fmap(map_id, total_cnt):
    smin, smax = 0.1, 0.9
    return smin + map_id * (smax - smin) / (total_cnt - 1)
