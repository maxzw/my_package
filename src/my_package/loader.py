import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from typing import Mapping, Sequence


class CustomDataset:
    def __init__(self, data: Sequence):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        return [x+"-1" for x in batch]


class GroupedDataloader:
    "A class that wraps a list of pytorch iterators, and makes it possible to iterate over them in a random order."
    def __init__(
        self,
        datasets: Sequence[CustomDataset],
        collate: bool = False,
        dataloader_kwargs: dict = {},
        shuffle_groups: bool = True,
    ) -> None:
        """Initialize the iterator.

        Args:
            datasets (Sequence[CustomDataset]): A list of datasets to iterate over.
            collate (bool): Whether to collate the data. Note that we call 'collate_fn' on the the datasets.
            dataloader_kwargs (dict): Keyword arguments to pass to the dataloader constructor.
            shuffle_groups (bool, optional): Whether to shuffle between groups. Defaults to True. If True,
                each group will be sampled weighed by the number of samples left in the group iterator.
        """
        self.dataloaders = [DataLoader(
            dataset,
            collate_fn=dataset.collate_fn if collate else None,
            **dataloader_kwargs,
        ) for dataset in datasets]
        self.iterators = [iter(dataloader) for dataloader in self.dataloaders]
        self.shuffle_groups = shuffle_groups
        self.counters = np.array([0 for _ in range(len(self.dataloaders))])
        self.lengths = np.array([len(dataloader) for dataloader in self.dataloaders])

    def __len__(self) -> int:
        "Return the total number of batches in the iterator."
        return sum([len(dataloader) for dataloader in self.dataloaders])

    def __iter__(self):
        "Return the iterator."
        return self

    def __next__(self):
        "Return the next element in the iterator."
        if not self.shuffle_groups:
            can_sample = self.counters < self.lengths
            if not can_sample.any():
                raise StopIteration
            dl_idx = np.where(can_sample)[0][0]
            self.counters[dl_idx] += 1
            return next(self.iterators[dl_idx])
        else:
            samples_to_go = self.lengths - (self.counters + 1)
            if not samples_to_go.any():
                raise StopIteration
            weights = samples_to_go / samples_to_go.sum()
            dl_idx = np.random.choice(len(self.dataloaders), p=weights)
            self.counters[dl_idx] += 1
            return next(self.iterators[dl_idx])


def get_data_loaders(
    batch_size: int,
    valid_size: float,
    num_workers: int = 0,
) -> Mapping[str, DataLoader]:
    """Return a dictionary of data loaders for the datasets.

    Args:
        batch_size (int): The batch size.
        split_size (Sequence[float]): The split size of the data.
        num_workers (int, optional): The number of workers to use for the dataloaders. Defaults to 0.
    
    Returns:
        Mapping[str, DataLoader]: A dictionary of data loaders for the datasets.
    """
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    num_train = int(len(train_data) * (1 - valid_size))
    num_valid = len(train_data) - num_train
    train_data, valid_data = torch.utils.data.random_split(train_data, [num_train, num_valid])
    return {
        key: DataLoader(value, batch_size=batch_size, num_workers=num_workers) for key, value in [
            ('train', train_data),
            ('valid', valid_data),
            ('test', test_data),
        ]
    }