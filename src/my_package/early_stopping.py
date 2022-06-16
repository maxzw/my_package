import logging


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after certain epochs.
    """

    def __init__(
        self, smaller_is_better: bool = True, patience: int = 5, min_delta: int = 0
    ) -> None:
        """Initialize the EarlyStopping class.

        Args:
            smaller_is_better (bool, optional): If True, the smaller the reported value is, the better. If False,
                the bigger the reported value is, the better. Defaults to True.
            patience (int, optional): how many epochs to wait before stopping when loss is
               not improving. Defaults to 5.
            min_delta (int, optional): minimum difference between new loss and old loss for
               new loss to be considered as an improvement. Defaults to 0.
        """
        self.smaller_is_better = smaller_is_better
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = None
        self.early_stop = False

    def is_better(self, value: float) -> bool:
        """Return True if the value is better than the best value."""
        if self.smaller_is_better:
            return value < self.best_value
        else:
            return value > self.best_value

    def report(self, value: float) -> None:
        """Report a new result.

        Args:
            value (float): The new result.
        """
        if self.best_value == None:
            self.best_value = value

        elif self.is_better(value):
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            logging.info(f"Early stopping counter: {self.counter}")
            if self.counter >= self.patience:
                self.early_stop = True
                logging.info("Early stopping: patience reached")

    def should_stop(self) -> bool:
        """Return True if the training should stop."""
        return self.early_stop

    def __repr__(self) -> str:
        return f"EarlyStopping(smaller_is_better={self.smaller_is_better}, patience={self.patience}, min_delta={self.min_delta})"
