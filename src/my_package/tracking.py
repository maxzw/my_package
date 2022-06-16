import wandb
from ctypes import cast
import logging
from pathlib import Path
from textwrap import dedent
from typing import Callable, Mapping, Optional, Sequence, Any
from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logging(
    folder_nesting: Optional[Sequence[str]] = [],
    file_name: Optional[str] = "log",
    return_path: bool = False,
) -> Optional[Path]:
    """Sets up the logging system and returns the log directory so other files can be saved there.

    Args:
        folder_nesting (Sequence[str], optional): The folder nesting to use. Defaults to [].
        file_name (str, optional): The file name to use. Defaults to 'log'.

    Returns:
        Path: The directory of the log file.
    """

    folder_path = Path("/".join(folder_nesting)).resolve()
    file_path = folder_path / (file_name + ".txt")
    Path(folder_path).mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)5s | %(levelname)5s | %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S",
        handlers=[
            logging.FileHandler(file_path),
            TqdmLoggingHandler(),  # With this, logging also prints to the terminal and works with tqdm
        ],
    )
    logging.info(f"Logging to {file_path}")
    if return_path:
        return folder_path


def setup_wandb(
    config: Mapping[str, Any] = {},
    wandb_kwargs: Optional[Mapping[str, Any]] = None,
) -> Optional[Callable[[Mapping[str, Any]], None]]:
    """Sets up the wandb logging system.

    Args:
        config (Mapping[str, Any]): The config to use.
        wandb_kwargs (Mapping[str, Any], optional): The kwargs to use. Defaults to None.

    Returns:
        Callable[Mapping[str, Any]]: The wandb logging to use.
    """
    wandb_run = wandb.init(
        project="my_project", config=config, reinit=True, **(wandb_kwargs or {})
    )

    def wandb_log_wrapper(result: Mapping[str, Any]) -> None:
        """Wrapper around Run.log."""
        wandb_run.log(dict(result))

    log_wrapper = wandb_log_wrapper

    return log_wrapper
