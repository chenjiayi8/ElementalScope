"""
Main file for the application.
"""

import json
import os
import sys
import time
from datetime import datetime
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QVBoxLayout
from skimage.color import rgb2gray
from skimage.draw import disk

from ElementalScope.io.data_tools import (
    ElementLoader,
    ElementWriter,
    HDF5Loader,
    HDF5Writer,
)
from ElementalScope.io.hdf5_tools import write_h5py_all_dataset

# # pylint: disable=unused-import
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
