import pathlib
import click
import pprint
import logging
from matplotlib import pyplot as plt
from matplotlib.style import use
import torch
from datetime import datetime
from my_package.early_stopping import EarlyStopping
from my_package.models import MLP

from my_package.training import train_iter

from .tracking import setup_logging, setup_wandb
from .hpo import optimize
from .loader import get_data_loaders

logger = logging.getLogger(__name__)

# ---- General options ----
# Model
# ...

# Training
option_device = click.option(
    "--device",
    type=click.Choice(["cpu", "cuda"]),
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Device to use for training.",
)
option_batch_size = click.option(
    "--batch-size",
    type=int,
    default=32,
    help="Batch size used.",
)
option_epochs = click.option(
    "-e", "--epochs", type=int, default=10000, help="The number of epochs to train."
)

# Logging
option_use_wandb = click.option(
    "-w", "--use-wandb", is_flag=True, default=False, help="Whether to use wandb."
)
option_model_path = click.option(
    "-m",
    "--model-path",
    type=pathlib.Path,
    default="./model.pt",
    help="The path to save the model to.",
)


@click.group()
def main():
    """This is the main entry point for the CLI."""
    pass


@main.command("train")
@option_device
@option_batch_size
@option_epochs
@click.option(
    "--optimizer",
    type=click.Choice(["SGD", "Adam", "Adagrad", "Adadelta", "RMSprop"]),
    default="Adam",
    help="Optimizer to use for training.",
)
@click.option(
    "--optimizer-kwargs",
    type=click.STRING,
    default=None,
    help="Optimizer kwargs to use for training.",
)
@click.option(
    "--learning-rate",
    type=float,
    default=0.001,
    help="Learning rate to use for training.",
)
@click.option(
    "--evaluation-frequency",
    type=int,
    default=1,
    help="The number of epochs between evaluations.",
)
@click.option(
    "--early-stopping",
    type=int,
    default=None,
    help="The number of epochs to wait before stopping training.",
)
@option_use_wandb
@click.option(
    "-s", "--save", is_flag=True, default=False, help="Whether to save the model."
)
@option_model_path
def train(
    device: str,
    batch_size: int,
    epochs: int,
    optimizer: str,
    optimizer_kwargs: dict,
    learning_rate: float,
    evaluation_frequency: int,
    early_stopping: int,
    use_wandb: bool,
    save: bool,
    model_path: pathlib.Path,
) -> None:
    """Train a model."""

    # Set up logging
    log_folder = setup_logging(
        folder_nesting=["logs", "train", datetime.now().strftime("%m-%d-%Y_%H-%M-%S")],
        file_name="training_log",
        return_path=True,
    )

    # Load data
    dataloaders, information = get_data_loaders(batch_size=batch_size, valid_size=0.2)

    # Define model
    model = MLP(
        input_size=information["num_features"],
        hidden_size=500,
        output_size=1,
    )
    logging.info(f"Model: {model}")

    # Define early stopper
    if early_stopping:
        early_stopper = EarlyStopping(
            patience=early_stopping,
        )

    # setup WandB
    config = {**information, **click.get_current_context().params}
    logging.info(f"Config: {pprint.pformat(config)}")
    if use_wandb:
        wandb_log = setup_wandb(config=config)

    # Train model
    for epoch, epoch_result in enumerate(
        train_iter(
            model=model,
            dataloaders=dataloaders,
            device=device,
            num_epochs=epochs,
            optimizer=optimizer,
            learning_rate=learning_rate,
            optimizer_kwargs=optimizer_kwargs,
            evaluation_frequency=evaluation_frequency,
        )
    ):
        # Log results
        logging.info(f"Epoch: {epoch:5}/{epochs:5}: {epoch_result}")
        if use_wandb:
            wandb_log(epoch_result)

        # Apply early stopping
        if early_stopping:
            early_stopper.report(epoch_result["valid"])
            if early_stopper.should_stop():
                logging.info(f"Early stopping at epoch {epoch}")
                break

    # Save model
    if save:
        torch.save(
            {"model": model.state_dict(), "config": config},
            model_path or log_folder / "model.pt",
        )


@main.command("evaluate")
@option_model_path
@torch.no_grad()
def optimize_cli(
    model_path: pathlib.Path,
) -> None:
    """Evaluate a model."""
    dataloaders, information = get_data_loaders(batch_size=100, valid_size=0.2)
    model = MLP(
        input_size=information["features"],
        hidden_size=500,
        output_size=1,
    )
    model.load_state_dict(torch.load(model_path)["model"])
    model.eval()
    valid_dataloader = dataloaders["valid"]
    true, pred = [], []
    for x, y in valid_dataloader:
        true.extend(torch.tensor(torch.e).pow(y + 12).cpu().numpy())
        pred.extend(torch.tensor(torch.e).pow(model(x) + 12).cpu().numpy())
    fig = plt.figure(figsize=(15, 15))
    ax1 = fig.add_subplot(111)
    ax1.set_title("True vs Predicted")
    ax1.scatter(list(range(0, len(true))), true, s=10, c="r", marker="o", label="True")
    ax1.scatter(
        list(range(0, len(pred))), pred, s=10, c="b", marker="o", label="Predicted"
    )
    plt.legend(loc="upper right")
    plt.savefig(str(model_path.parent / "true_vs_predicted.png"), bbox_inches="tight")
    plt.show()


@main.command("optimize")
@click.option(
    "-p", "--pruner", type=str, default=None, help="The pruner to use for optimization."
)
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
