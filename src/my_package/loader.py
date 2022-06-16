import pathlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Mapping, Sequence

from my_package.preprocessing import preprocess_data


class CustomDataset:
    def __init__(self, data_path: pathlib.Path):
        self.X, self.y = preprocess_data(data_path)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def collate_fn(self, batch):
        return torch.stack([x for x, _ in batch]), torch.stack([y for _, y in batch])


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
        self.dataloaders = [
            DataLoader(
                dataset,
                collate_fn=dataset.collate_fn if collate else None,
                **dataloader_kwargs,
            )
            for dataset in datasets
        ]
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
    data = CustomDataset("datasets/train.csv")

    num_train = int(len(data) * (1 - valid_size))
    num_valid = len(data) - num_train

    train_data, valid_data = torch.utils.data.random_split(data, [num_train, num_valid])

    dataloaders = {
        split: DataLoader(
            data,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=(split == "train"),
        )
        for split, data in [
            ("train", train_data),
            ("valid", valid_data),
        ]
    }

    information = {
        "num_features": train_data.dataset.X.shape[1],
        "num_samples": len(data),
        "num_train_samples": num_train,
        "num_val_samples": num_valid,
    }

    return dataloaders, information
