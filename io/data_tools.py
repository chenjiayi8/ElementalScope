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
        """Set the task for the thread."""
        self.folder_name = folder_name
        self.full_path = full_path

    def run(self):
        """Run the thread."""
        if not self.full_path:
            return
        data = read_h5py_all_dataset(self.full_path)
        self.data_loaded.emit((self.folder_name, data))


class ElementLoader(QThread):
    """This class loads the data from the element csv file."""

    data_loaded = pyqtSignal(tuple)

    def __init__(self):
        super().__init__()
        self.folder_name = None
        self.full_path = None

    def set_task(self, folder_name, full_path):
        """Set the task for the thread."""
        self.folder_name = folder_name
        self.full_path = full_path

    def run(self):
        """Run the thread."""
        if not self.full_path:
            return
        try:
            (
                _,
                _,
                data_container,
                resolution,
                _,
            ) = read_element_data(self.full_path)
            data_container["resolution"] = resolution
        except ValueError as error:
            self.data_loaded.emit((self.folder_name, error))
            return
        self.data_loaded.emit((self.folder_name, data_container))


class HDF5Writer(QThread):
    """This class writes the data to the HDF5 file."""

    data_written = pyqtSignal(str)

    def __init__(self, writer_task):
        super().__init__()
        self.writer_task = writer_task

    def run(self):
        """Run the thread."""
        if not self.writer_task:
            return
        result_path = self.writer_task["result_path"]
        data_container = self.writer_task["result_data"]
        write_h5py_all_dataset(result_path, data_container)
        self.data_written.emit(self.writer_task["task_name"])


class ElementWriter(QThread):
    """This class writes the data to the element csv file."""

    data_written = pyqtSignal(str)

    def __init__(self, writer_task):
        super().__init__()
        self.writer_task = writer_task

    def run(self):
        """Run the thread."""
        if not self.writer_task:
            return

        for field, value in self.writer_task["result_data"].items():
            write_atom_data(
                self.writer_task["task_path"],
                self.writer_task["task_name"],
                field,
                self.writer_task["resolution"],
                value,
            )
        self.data_written.emit(self.writer_task["task_name"])
