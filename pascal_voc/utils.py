import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler


def make_dataloaders(dataset, train_batch_size=1, val_batch_size=1, validation_split=0.1, shuffle_dataset=True):
    random_seed = 42

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=train_batch_size,
                                                   sampler=train_sampler)
    validation_dataloader = torch.utils.data.DataLoader(dataset, batch_size=val_batch_size,
                                                        sampler=valid_sampler)
    return train_dataloader, validation_dataloader
