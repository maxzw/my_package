"""A module for preprocessing."""

import pathlib
from typing import Tuple
import pandas
import torch
import torch.nn.functional as F


def preprocess_data(
    data_path: pathlib.Path,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """Preprocess the data.

    Args:
        data_path (pathlib.Path): Path to the data.

    Returns:
        Tuple[torch.FloatTensor, torch.FloatTensor]: The preprocessed data.
    """
    # Read data
    df = pandas.read_csv(data_path).drop(["Id"], axis=1)

    # Take out target
    targets = torch.tensor(df.pop("SalePrice").to_numpy(), dtype=torch.float).log()-12

    # Remove columns with a lot of missing data
    df.dropna(
        axis=1,
        how="all",
        thresh=int(0.8 * len(df)),
        inplace=True,
    )

    # Split data into numerical and categorical
    numerical_df = df.select_dtypes(include=["float64", "int64"])
    # categorical_df = df.select_dtypes(include=["object"])

    # Impute numerical values with the mean
    for column in numerical_df.columns:
        if column == "GarageYrBlt":  # Impute GarageYrBlt with building year of house
            numerical_df[column].fillna(numerical_df["YearBuilt"], inplace=True)
        else:
            numerical_df[column].fillna(numerical_df[column].mean(), inplace=True)

    # # Impute categorical values with the mode
    # for column in categorical_df.columns:
    #     categorical_df[column].fillna(categorical_df[column].mode()[0], inplace=True)

    # Make numerical values tensor
    numerical_t = torch.tensor(numerical_df.to_numpy(), dtype=torch.float)

    # Normalize numerical values
    numerical_t = F.normalize(numerical_t, dim=0)

    # # Make one-hot categorical values
    # categorical_df = pandas.get_dummies(categorical_df)

    # # Make categorical values tensor
    # categorical_t = torch.tensor(categorical_df.to_numpy(), dtype=torch.float)

    # # Combine all features into a single tensor
    # features = torch.cat((numerical_t, categorical_t), dim=1)

    return numerical_t, targets
