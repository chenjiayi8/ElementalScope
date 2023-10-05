"""H5py module for ElementalScope."""

import h5py
import numpy as np

DEFAULT_COMPRESSION_OPTS = 3


def read_h5py_all_dataset(file_path):
    """Load all h5py datasets."""
    with h5py.File(file_path, "r") as file:
        datasets = {}
        for key, value in file.items():
            # Check if the item is a dataset
            if isinstance(value, h5py.Dataset):
                datasets[key] = value[...]  # Reads the entire dataset
        return datasets

