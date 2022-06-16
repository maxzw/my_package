import pickle
from types import NoneType
from typing import Any, Mapping, Tuple
from class_resolver import Hint
from class_resolver.contrib.optuna import pruner_resolver, sampler_resolver
import optuna
from optuna.pruners import BasePruner
from optuna.samplers import BaseSampler
from dataclasses import dataclass

from .tracking import setup_logging, setup_wandb


@dataclass
class Objective:
    """An objective for optuna that is used to optimize the hyperparameters of a model."""

    def __init__(
        self,
        use_wandb: bool,
        dataset: str,
        metric: str,
    ) -> None:
        """Initialize the objective.

        Args:
            use_wandb (bool): Whether to use wandb.
            dataset (str): The dataset to use.
            metric (str): The metric to use.
        """
        self.use_wandb = use_wandb
        self.dataset = dataset
        self.metric = metric

    def __call__(self, trial: optuna.Trial) -> float:
        """Evaluate the objective function."""

        activation = trial.suggest_categorical(
            "activation", ["relu", "elu", "selu", "tanh"]
        )
        optimizer = trial.suggest_categorical(
            "optimizer", ["sgd", "adam", "adagrad", "adadelta", "rmsprop"]
        )
        # ...

        # initialize model

        # train the model
        # ...

        raise NotImplementedError()


def optimize(
    dataset: str,
    use_wandb: bool,
    metric: str,
    sampler_str: Hint[BasePruner] = None,
    pruner_str: Hint[BaseSampler] = None,
    save_study: bool = False,
    save_study_path: str = None,
) -> Tuple[Mapping[str, Any], float]:
    """Optimize the hyperparameters of a model.

    Args:
        dataset (str): The dataset to use for training.
        use_wandb (bool): Whether to use wandb.
        save_study (bool): Whether to save the study.

    Returns:
        Tuple[Mapping[str, Any], float]: The best hyperparameters and the best objective function value.
    """
    objective = Objective(
        use_wandb,
        dataset,
        metric,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler_resolver.make(sampler_str),
        pruner=pruner_resolver.make(pruner_str),
    )
    study.optimize(objective, n_trials=100)
    if save_study:
        pickle.dump(study, open("study.pkl", "wb"))
    return study.best_params, study.best_value
