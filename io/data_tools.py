""" This module contains the classes for loading and writing data. """
from PyQt5.QtCore import QThread, pyqtSignal

from ElementalScope.io.hdf5_tools import (
    read_h5py_all_dataset,
    write_h5py_all_dataset,
)
from ElementalScope.utils.element import read_element_data, write_atom_data


class HDF5Loader(QThread):
    """This class loads the data from the HDF5 file."""

    data_loaded = pyqtSignal(tuple)

    def __init__(self):
        super().__init__()
        self.folder_name = None
        self.full_path = None

    def set_task(self, folder_name, full_path):
        self.folder_name = folder_name
        self.full_path = full_path

    def run(self):
        if not self.full_path:
            return
        data = read_h5py_all_dataset(self.full_path)
        self.data_loaded.emit((self.folder_name, data))


