import click
import pprint
import logging
from time import sleep
import numpy as np
from tqdm import tqdm
from datetime import datetime

from my_package.training import train_iter

from .tracking import setup_logging, setup_wandb
from .hpo import optimize
from .loader import get_data_loaders

logger = logging.getLogger(__name__)

# General options
option_batch_size = click.option('-b', '--batch_size', default=32, help='The batch size to use.')
option_epochs = click.option('-e', '--epochs', default=10, help='The number of epochs to train.')
option_use_wandb = click.option('--use_wandb', is_flag=True, default=False, help='Whether to use wandb.')


@click.group()
def main():
    """This is the main entry point for the CLI."""
    pass


@main.command("train")
@option_batch_size
@option_epochs
@option_use_wandb
@click.option('-e', '--epochs', default=10, help='The number of epochs to train for.')
def train(
    batch_size: int,
    epochs: int,
    use_wandb: bool,
) -> None:
    """Train a model."""

    # set up logging
    log_folder = setup_logging(
        folder_nesting=["logs", "train", datetime.now().strftime("%m-%d-%Y_%H-%M-%S")],
        file_name="train",
        return_path=True
    )
    logger.info(f"Using params: {pprint.pformat(locals())}")

    dataloaders = get_data_loaders(batch_size=batch_size, valid_size=0.2)
    model = MyModel()

    for epoch, epoch_result in enumerate(train_iter(
        model=model,
        dataloaders=dataloaders,
    )):
        logger.info(f"Epoch: {epoch:5}/{epochs:5}: {epoch_result}")


@main.command("optimize")
@click.option(
    "-p", "--pruner", type=str, default=None, help="The pruner to use for optimization."
)
@option_dataset
@option_batch_size
@option_use_wandb
def optimize_cli(
    pruner: str,
    dataset: str,
    batch_size: int,
    use_wandb: bool,
) -> None:
    """Optimize a model."""
    result = optimize(pruner, dataset, batch_size, use_wandb)
    print(f"Optimized hyperparameters: {result}")