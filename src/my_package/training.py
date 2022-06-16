import logging
import optuna
import torch
from torch.utils.data import DataLoader
from typing import Any, Mapping, Optional, Sequence
from class_resolver import ClassResolver, Hint
import wandb

from my_package.tracking import setup_wandb
from my_package.utils import get_from_nested_dict

optimizer_resolver = ClassResolver(
    classes = [
        torch.optim.SGD,
        torch.optim.Adam,
        torch.optim.Adagrad,
        torch.optim.Adadelta,
        torch.optim.RMSprop
    ],
    base=torch.optim.Optimizer,
    default=torch.optim.Adam,
)


def _train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"Epoch {i}/{len(dataloader)}: {loss.item()}")


def train_iter(
    model,
    dataloaders: Mapping[str, DataLoader],
    loss_function: Hint[Any] = None,
    loss_function_kwargs: Optional[Mapping[str, Any]] = None,
    optimizer: Hint[torch.optim.Optimizer] = None,
    optimizer_kwargs: Optional[Mapping[str, Any]] = None,
    num_epochs: int = 1,
    evaluation_frequency: int = 1,
    use_wandb: bool = True,
    early_stopper: Optional[Hint[Any]] = None,
    early_stopper_kwargs: Optional[Mapping[str, Any]] = None,
    early_stopper_metric: Sequence[str] = None,
    trial: Optional[optuna.Trial] = None,
    trial_metric: Sequence[str] = None,
):

    # setup WandB
    if use_wandb: wandb_log = setup_wandb()

    optimizer_instance = optimizer_resolver.make(
        optimizer, 
        optimizer_kwargs,
        model.parameters()
    )

    # Training loop
    try:
        for epoch in range(1, num_epochs):
            result = {}
            result["train"] = _train_epoch(model, dataloaders["train"], optimizer, criterion, device)
            
            # Evaluate
            if (epoch+1) % evaluation_frequency == 0:
                for split in dataloaders.keys():
                    if split == "train":
                        continue
                    result[split] = eval_epoch(model, dataloaders[split], criterion, device)
            
            # Log
            if use_wandb: wandb_log(result)

            # If needed, handle pruning based on the intermediate objective value
            objective_value = get_from_nested_dict(result, trial_metric)

            # If hpo, return the objective value
            if trial is not None:
                trial.report(objective_value, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                yield objective_value
            
            # If regular training, return the results
            if (trial is None) and (early_stopper is not None):
                early_stopper(objective_value)
                if early_stopper.early_stop:
                    break
            
            yield result

    except RuntimeError as error:
        logging.fatal(f"RuntimeError: {error}")
        exit_code = -1
    
    finally:
        if use_wandb: wandb.finish(exit_code)
        return None