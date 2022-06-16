import logging
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Any, Mapping, Optional
from class_resolver import ClassResolver, Hint

from my_package.models import MLP

logger = logging.getLogger(__name__)

optimizer_resolver = ClassResolver(
    classes=[
        torch.optim.SGD,
        torch.optim.Adam,
        torch.optim.Adagrad,
        torch.optim.Adadelta,
        torch.optim.RMSprop,
    ],
    base=torch.optim.Optimizer,
    default=torch.optim.Adam,
)


def _train_epoch(
    model: MLP,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    model.train()
    running_loss = torch.tensor(0.0)
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.mse_loss(outputs.squeeze(), targets, reduction="sum")
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)


@torch.no_grad()
def _eval_epoch(model: MLP, dataloader: DataLoader, device: torch.device):
    model.eval()
    running_loss = torch.tensor(0.0)
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = F.mse_loss(outputs, targets, reduction="sum")
        running_loss += loss.item() * inputs.size(0)
    return loss.item() / len(dataloader.dataset)


def train_iter(
    model: MLP,
    device: torch.device,
    dataloaders: Mapping[str, DataLoader],
    optimizer: Hint[torch.optim.Optimizer] = None,
    learning_rate: float = 0.001,
    optimizer_kwargs: Optional[Mapping[str, Any]] = None,
    num_epochs: int = 1,
    evaluation_frequency: int = 1,
):

    # Define optimizer
    optimizer_instance = optimizer_resolver.make(
        optimizer,
        optimizer_kwargs,
        lr=learning_rate,
        params=model.parameters(),
    )
    logging.info(f"Optimizer: {optimizer_instance}")

    # Training loop
    for epoch in range(1, num_epochs):
        result = {}
        result["train"] = _train_epoch(
            model,
            dataloaders["train"],
            optimizer_instance,
            device=device,
        )

        # Evaluate
        if (epoch + 1) % evaluation_frequency == 0:
            for split in dataloaders.keys():
                if split == "train":
                    continue
                result[split] = _eval_epoch(
                    model,
                    dataloaders[split],
                    device=device,
                )

        yield result
