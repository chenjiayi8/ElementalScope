"""
Main file for the application.
"""

import json
import os
import sys
from datetime import datetime
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QVBoxLayout
from skimage.color import rgb2gray
from skimage.draw import disk

from ElementalScope.io.data_tools import (
    ElementLoader,
    ElementWriter,
    HDF5Loader,
    HDF5Writer,
)

# pylint: disable=unused-import
from ElementalScope.ui.generated import resources_rc
from ElementalScope.ui.generated.mainwindow_ui import Ui_MainWindow
from ElementalScope.utils.debounce import debounce
from ElementalScope.utils.image import imshowpair
from ElementalScope.utils.matrix import add_small_to_big_matrix_2d_periodically
from ElementalScope.utils.string import get_common_prefix


class MyMainWindow(QMainWindow, Ui_MainWindow):
    """
    Main window class.
    """

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self._root = None  # root folder
        self._folders = {}  # working folders map
        self._data_loaders = {}  # data loaders
        self._hdf5_writer = None  # hdf5 writer
        self._element_writer = None  # element writer
        self._loaded_folders = None  # count of loaded folders
        self._tasks = {}  # saved tasks
        self._data_container = {}  # data container
        self._add_x = None
        self._add_y = None
        self._comparison_image = None  # compared image
        self._resolution = None  # resolution
        self.connect_callbacks()
        self.placeholder()

    def get_root(self):
        """Get the root folder."""
        return self._root

    def connect_callbacks(self):
        """Connect the callbacks"""
        self.connect_control_panel()
        self.connect_move_buttons()

    def connect_control_panel(self):
        """Connect the callbacks for the control panel."""
        self.pushButton_choose_folder.clicked.connect(self.choose_folder)
        self.pushButton_compare.clicked.connect(self.compare)
        self.pushButton_save.clicked.connect(self.save)
        self.pushButton_stitch.clicked.connect(self.stitch)
        self.pushButton_reset.clicked.connect(self.reset)
        self.pushButton_exit.clicked.connect(self.close)
        self.comboBox_left.currentTextChanged.connect(
            self.update_element_choices
        )
        self.comboBox_right.currentTextChanged.connect(
            self.update_element_choices
        )
        self.comboBox_task.currentTextChanged.connect(self.restore_task)

    def connect_move_buttons(self):
        """Connect the callbacks for the move buttons."""
        self.pushButton_x_m.clicked.connect(partial(self.move_x, -1))
        self.pushButton_x_p.clicked.connect(partial(self.move_x, 1))
        self.pushButton_x_mm.clicked.connect(partial(self.move_x, -10))
        self.pushButton_x_pp.clicked.connect(partial(self.move_x, 10))
        self.pushButton_y_m.clicked.connect(partial(self.move_y, -1))
        self.pushButton_y_p.clicked.connect(partial(self.move_y, 1))
        self.pushButton_y_mm.clicked.connect(partial(self.move_y, -10))
        self.pushButton_y_pp.clicked.connect(partial(self.move_y, 10))

    def move_x(self, value):
        """Move the image in the x direction."""
        d_x = int(self.lineEdit_dx.text()) + value
        self.lineEdit_dx.setText(str(d_x))
        self.compare()

    def move_y(self, value):
        """Move the image in the y direction."""
        d_y = int(self.lineEdit_dy.text()) + value
        self.lineEdit_dy.setText(str(d_y))
        self.compare()

    def hint(self, message):
        """Show the message in the hint box."""
        time_prefix = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S.%f")
        time_prefix = time_prefix[:-3] + "$ "
        message = time_prefix + message
        self.textBrowser_hint.append(message)

    def initialise(self):
        """Initialise the application."""
        self._folders.clear()
        self._loaded_folders = 0
        self.read_inputs()
        self.read_outputs()
        self.load_data()

    def read_subfolders(self, parent):
        """Read the subfolders in the parent folder."""
        # List all entries in the parent directory
        entries = os.listdir(parent)

        # Filter out non-directory entries
        subfolders = [
            entry
            for entry in entries
            if os.path.isdir(os.path.join(parent, entry))
        ]

        for folder_name in subfolders:
            if folder_name == "Output":
                continue

            if folder_name not in self._folders:
                full_path = os.path.join(parent, folder_name)
                relative_path = os.path.relpath(full_path, self._root)
                self._folders[folder_name] = relative_path

        return subfolders

    def read_inputs(self):
        """Read the input folder."""
        self.read_subfolders(self._root)
        os.makedirs(os.path.join(self._root, "Output"), exist_ok=True)

    def read_outputs(self):
        """Read the output folder."""
        output_folder = os.path.join(self._root, "Output")
        output_folders = self.read_subfolders(output_folder)
        output_folders = sorted(output_folders)
        for folder_name in output_folders:
            result_file = f"{folder_name}.json"
            result_path = os.path.join(output_folder, folder_name, result_file)

            with open(result_path, mode="r", encoding="utf-8") as file:
                self._tasks[folder_name] = json.load(file)

        self.comboBox_task.clear()
        self.comboBox_task.addItems(self._tasks.keys())
        self.comboBox_task.addItem("New")

    def load_data(self):
        """Load the data from the folders."""
        self._data_container.clear()
        folder_names = list(self._folders.keys())

        for folder_name in folder_names:
            # self.hint(f"Loading from folder: {folder_name}", 3)
            folder_path = os.path.join(self._root, self._folders[folder_name])
            result_path = os.path.join(folder_path, f"{folder_name}.h5")
            if os.path.exists(result_path):
                # self.hint(f"Loading from folder: {folder_name}")
                data_loader = HDF5Loader()
                data_loader.set_task(folder_name, result_path)
                data_loader.data_loaded.connect(self.on_hdf5_data_loaded)
                self._data_loaders[folder_name] = data_loader
                self._data_loaders[folder_name].start()

            else:
                data_loader = ElementLoader()
                data_loader.set_task(folder_name, folder_path)
                data_loader.data_loaded.connect(self.on_element_data_loaded)
                self._data_loaders[folder_name] = data_loader
                self._data_loaders[folder_name].start()

    def on_hdf5_data_loaded(self, data_tuple):
        """Callback when the hdf5 data is loaded."""
        task_name, data = data_tuple
        self._data_container[task_name] = data
        self._loaded_folders += 1
        if self._loaded_folders == len(self._folders):
            self.on_all_data_loaded()

    def on_element_data_loaded(self, data_tuple):
        """Callback when the element data is loaded."""
        task_name, data_container = data_tuple
        if isinstance(data_container, ValueError):
            self.hint(f"Error: {data_container}")
            self._loaded_folders += 1
            if self._loaded_folders == len(self._folders):
                self.on_all_data_loaded()
            return
        self._data_container[task_name] = data_container
        folder_path = os.path.join(self._root, self._folders[task_name])
        result_path = os.path.join(folder_path, f"{task_name}.h5")
        hdf5_writer_task = {
            "result_path": result_path,
            "result_data": data_container,
            "task_name": task_name,
        }
        self.pushButton_stitch.setEnabled(False)
        self.pushButton_stitch.setText("Writing ...")
        self._hdf5_writer = HDF5Writer(hdf5_writer_task)
        self._hdf5_writer.data_written.connect(self.on_hdf5_data_written)
        self._hdf5_writer.start()
        self._loaded_folders += 1
        if self._loaded_folders == len(self._folders):
            self.on_all_data_loaded()

    def on_hdf5_data_written(self, task_name):
        """Callback when the hdf5 data is written."""
        self.hint(f"{task_name} HDF5 data written!")
        self._hdf5_writer = None
        if self._element_writer is None:
            self.pushButton_stitch.setEnabled(True)
            self.pushButton_stitch.setText("Stitch")

    def on_element_data_written(self, task_name):
        """Callback when the element data is written."""
        self.hint(f"{task_name} Element data written!")
        self._element_writer = None
        if self._hdf5_writer is None:
            self.pushButton_stitch.setEnabled(True)
            self.pushButton_stitch.setText("Stitch")

    def on_all_data_loaded(self):
        """Callback when all the data is loaded."""
        self.pushButton_choose_folder.setEnabled(True)
        self.pushButton_choose_folder.setText("Choose Folder")
        self.hint("All folders loaded!")
        self.update_image_choices()
        self.update_element_choices()
        tasks = sorted(self._tasks.keys())
        tasks = [task for task in tasks if task in self._data_container]
        self.comboBox_task.setCurrentText(tasks[0])
        self.restore_task()
        self._data_loaders.clear()

