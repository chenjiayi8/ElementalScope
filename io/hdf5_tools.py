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


def write_h5py_all_dataset(file_path, datasets):
    """Write all h5py datasets."""
    with h5py.File(file_path, "a") as file:
        for key, value in datasets.items():
            if key in file:
                del file[key]  # Delete the old dataset if it exists
            if np.isscalar(value):
                file.create_dataset(key, data=value)
            else:
                file.create_dataset(
                    key,
                    data=value,
                    compression="gzip",
                    compression_opts=DEFAULT_COMPRESSION_OPTS,
                )


def read_h5py_dataset(path, dataset_name):
    """Load h5py dataset."""
    with h5py.File(path, "r") as file:
        return file[dataset_name][:]


def read_h5py_dataset_as_array(path, dataset_name):
    """Load h5py dataset as array."""
    with h5py.File(path, "r") as file:
        return np.array(file[dataset_name][:])


def write_h5py_dataset(path, dataset_name, data):
    """Write h5py dataset."""
    with h5py.File(path, "a") as file:
        if dataset_name in file:
            del file[dataset_name]  # Delete the old dataset if it exists
        file.create_dataset(
            dataset_name,
            data=data,
            compression="gzip",
            compression_opts=DEFAULT_COMPRESSION_OPTS,
        )
