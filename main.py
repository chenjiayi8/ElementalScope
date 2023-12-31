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
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.QtSvg import QSvgRenderer
from PyQt5.QtWidgets import (
    QApplication,
    QDesktopWidget,
    QFileDialog,
    QMainWindow,
    QSplashScreen,
    QVBoxLayout,
)
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

    def __init__(self, splash):
        super().__init__()
        self.splash = splash
        self.setupUi(self)
        self._root = None  # root folder
        self._folders = {}  # working folders map
        self._data_loaders = {}  # data loaders
        self._hdf5_writer = None  # hdf5 writer
        self._element_writer = None  # element writer
        self._loaded_folders = None  # count of loaded folders
        self._tasks = {}  # saved tasks
        self._data_container = {}  # data container
        self.left_out = None  # left image for comparison
        self._add_x = None
        self._add_y = None
        self._comparison_image = None  # compared image
        self._resolution = None  # resolution
        self.init_ui()
        self.connect_callbacks()
        self.placeholder()
        self.placeholder_removed = False

    def init_ui(self):
        """Initialise the user interface elements."""
        # Create a Matplotlib figure and a plot
        self.figure, self.axis = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        # Embed the Matplotlib canvas inside the mainwidget
        plot_layout = QVBoxLayout(self.mainwidget)
        plot_layout.addWidget(self.canvas)

        # Embed the Matplotlib toolbar inside the placeholder
        toolbar_layout = QVBoxLayout(self.placeholder_matplotlib_menu)
        self.nav_toolbar = NavigationToolbar(self.canvas, self)
        toolbar_layout.addWidget(self.nav_toolbar)

    def show_maximized(self):
        """Show the main window."""
        super().showMaximized()
        self.splash.finish(self)

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
        self.comboBox_element.currentTextChanged.connect(self.update_left_out)

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

    def choose_folder(self):
        """Choose the root folder."""
        self._root = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not self._root:
            return

        _, folder_name = os.path.split(self._root)

        self.hint(f"Loading data from: {folder_name}")
        self.pushButton_choose_folder.setEnabled(False)
        self.pushButton_choose_folder.setText("Loading ...")
        self.initialise()

    def placeholder(self):
        """Show a placeholder image."""
        self.axis.clear()
        # Create two images with circles
        size = (256, 256)
        circle_a = np.zeros(size)
        circle_b = np.zeros(size)

        # Circle in the center for A
        row_indices, col_indices = disk(
            (size[0] // 2, size[1] // 2), radius=50
        )
        circle_a[row_indices, col_indices] = 1

        # Circle in the top-left for B
        row_indices, col_indices = disk(
            (size[0] // 4, size[1] // 4), radius=50
        )
        circle_b[row_indices, col_indices] = 1

        # Use imshowpair with the falsecolor method
        output = imshowpair(circle_a, circle_b, method="falsecolor")
        self.axis.imshow(output)
        self.axis.axis("off")
        self.figure.tight_layout()
        self.canvas.draw()

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
            folder_path = os.path.join(self._root, self._folders[folder_name])
            result_path = os.path.join(folder_path, f"{folder_name}.h5")
            if os.path.exists(result_path):
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

    def update_image_choices(self):
        """Update the image choices."""
        choices = list(self._data_container.keys())
        choices = sorted(choices)
        self.comboBox_left.clear()
        self.comboBox_right.clear()
        self.comboBox_left.addItems(choices)
        self.comboBox_right.addItems(choices)
        self._resolution = self._data_container[choices[0]]["resolution"]

    def update_element_choices(self):
        """Update the element choices."""
        if (self.comboBox_left.currentText() == "") or (
            self.comboBox_right.currentText() == ""
        ):
            return
        left = self._data_container[self.comboBox_left.currentText()]
        right = self._data_container[self.comboBox_right.currentText()]
        fields = list(set(left.keys()) & set(right.keys()))
        fields = [field for field in fields if field != "resolution"]
        self.comboBox_element.clear()
        self.comboBox_element.addItems(fields)
        if "Grey" in fields:
            self.comboBox_element.setCurrentText("Grey")

        self.left_or_right_changed()

    def on_left_changed(self):
        """Callback when the left image is changed."""
        self.update_element_choices()
        self.left_or_right_changed()
        self.update_left_out()

    def update_left_out(self):
        """Update the left image for comparison."""
        field = self.comboBox_element.currentText()
        left_name = self.comboBox_left.currentText()
        if field == "":
            return
        if left_name not in self._data_container:
            return
        left_data = self.get_left_data(field)
        self.left_out = self.precondition_left(left_data.copy())

    def left_or_right_changed(self):
        """Callback when the left or right image is changed."""
        task_name = self.get_task_name()
        if self.comboBox_task.currentText() != task_name:
            if task_name in self._tasks:
                self.comboBox_task.setCurrentText(task_name)
            else:
                self.comboBox_task.setCurrentText("New")

    def get_left_data(self, field):
        """Get the left data."""
        left_cont = self._data_container[self.comboBox_left.currentText()]
        if self.checkBox_transpose.isChecked():
            return left_cont[field].T
        return left_cont[field]

    def get_right_data(self, field):
        """Get the right data."""
        right_cont = self._data_container[self.comboBox_right.currentText()]
        if self.checkBox_transpose.isChecked():
            return right_cont[field].T
        return right_cont[field]

    def precondition_left(self, left_data):
        """Precondition the left data and return the left data with the offset."""
        rows, cols = left_data.shape
        mask = np.zeros((rows * 3, cols * 3))
        core_x = round(cols * 1.5)
        core_y = round(rows * 1.5)
        left_out, _, _ = add_small_to_big_matrix_2d_periodically(
            mask, left_data, core_x, core_y
        )
        return left_out

    def precondition_right(self, right_data):
        """Precondition the right data and return the right data with the offset."""
        mask = np.zeros(self.left_out.shape)
        percent_x = self.horizontalSlider_x.value() / 100
        percent_y = self.horizontalSlider_y.value() / 100
        d_x = int(self.lineEdit_dx.text())
        d_y = int(self.lineEdit_dy.text())
        self._add_x = round(self.left_out.shape[1] * percent_x + d_x)
        self._add_y = round(self.left_out.shape[0] * percent_y + d_y)
        right_out, _, _ = add_small_to_big_matrix_2d_periodically(
            mask, right_data, self._add_x, self._add_y
        )
        self._add_x -= round(self.left_out.shape[1] / 2)
        self._add_y -= round(self.left_out.shape[0] / 2)
        return right_out

    def get_right_out(self, field):
        """Get the right output."""
        right_data = self.get_right_data(field)
        right_out = self.precondition_right(right_data.copy())
        return right_out

    def get_boundary(self, img, threshold_value):
        """
        This function finds the boundary of a binary mask of the dark border \
        in the input image.

        Args:
        - img: A 2D or 3D numpy array of the input image.
        - threshold_value: An integer representing the threshold value for \
        the grayscale image.

        Returns:
        - boundary: A list of four integers representing the left, right, \
        top, and bottom indices of the boundary.
        """
        if len(img.shape) == 3:
            gray_img = rgb2gray(img)
        elif len(img.shape) == 2:
            gray_img = img
        else:
            raise ValueError("Input is not gray or rgb image")

        bin_img = gray_img > threshold_value

        # rows, cols = bin_img.shape
        row_sum = np.sum(bin_img, axis=0)
        col_sum = np.sum(bin_img, axis=1)
        left_idx, right_idx = self.find_border(row_sum)
        top_idx, bottom_idx = self.find_border(col_sum)
        boundary = [left_idx, right_idx, top_idx, bottom_idx]
        return boundary

    def find_border(self, sums):
        """
        This function finds the left or top index and the right or bottom index of the border.

        Args:
        - sums: A 1D numpy array of the row or column sums.

        Returns:
        - idx1: An integer representing the left or top index of the border.
        - idx2: An integer representing the right or bottom index of the border.
        """
        idx1 = -1
        idx2 = -1
        for i in range(1, len(sums)):
            if sums[i] != 0 and idx1 == -1:
                idx1 = i - 1
                break

        for i in range(len(sums) - 1, 0, -1):
            if sums[i] != 0 and idx2 == -1:
                idx2 = i + 1
                break

        return idx1, idx2

    def compare(self):
        """Compare the two images."""
        self.hint("Comparing ...")
        field = self.comboBox_element.currentText()
        right_out = self.get_right_out(field)
        self._comparison_image = imshowpair(
            self.left_out / self.left_out.max(),
            right_out / right_out.max(),
            method="falsecolor",
        )
        self.hint(f"Comparing done! (dx, dy) = ({self._add_x}, {self._add_y})")
        self.plot_diff()

    @debounce(1)
    def plot_diff(self):
        """Plot the difference image."""
        if self.placeholder_removed:
            # Get current axis limits
            xlim = self.axis.get_xlim()
            ylim = self.axis.get_ylim()
        else:
            ylim = [0, self._comparison_image.shape[0]]
            xlim = [0, self._comparison_image.shape[1]]
        self.axis.clear()
        self.axis.imshow(self._comparison_image)
        self.axis.axis("off")
        self.axis.set_xlim(xlim)
        self.axis.set_ylim(ylim)
        self.canvas.draw()
        self.placeholder_removed = True

    def save(self):
        """Save the offset and the transpose flag
        to the result file."""
        task_name = self.get_task_name()
        if task_name is None:
            self.hint("Error: The two images are the same!")
            return -1
        task_path = os.path.join(self._root, "Output", task_name)
        if not os.path.exists(task_path):
            os.makedirs(task_path)
        result_path = os.path.join(task_path, f"{task_name}.json")

        result = {
            "name": task_name,
            "element": self.comboBox_element.currentText(),
            "left": self.comboBox_left.currentText(),
            "right": self.comboBox_right.currentText(),
            "addX": self._add_x,
            "addY": self._add_y,
            "transpose": self.checkBox_transpose.isChecked(),
        }
        with open(result_path, mode="w", encoding="utf-8") as file:
            file.write(json.dumps(result, indent=4))
        self.hint(f"Saved to {result_path}")
        if task_name not in self._tasks:
            self.comboBox_task.addItem(task_name)
            self.comboBox_task.setCurrentText(task_name)

        self._tasks[task_name] = result
        return 0

    def stitch(self):
        """Stitch the two images."""
        exit_code = self.save()
        if exit_code != 0:
            return
        self.pushButton_stitch.setEnabled(False)
        self.pushButton_stitch.setText("Stitching ...")
        boundary = self.get_boundary(self._comparison_image, 0)
        fields = [
            self.comboBox_element.itemText(i)
            for i in range(self.comboBox_element.count())
        ]
        task_name = self.get_task_name()
        task_path = os.path.join(self._root, "Output", task_name)
        result_data = {}
        for field in fields:
            right_out = self.get_right_out(field)
            temp_out = self.left_out + right_out
            temp_out[temp_out > self.left_out] = right_out[
                temp_out > self.left_out
            ]
            final_out = temp_out[
                boundary[2] : boundary[3], boundary[0] : boundary[1]
            ]
            if self.checkBox_transpose.isChecked():
                final_out = final_out.T
            result_data[field] = final_out

        self._data_container[task_name] = result_data
        self.update_image_choices()
        self.comboBox_task.setCurrentText(task_name)

        self.hint("Saving hdf5 and element data ...")
        # create hdf5 writing task
        result_path = os.path.join(task_path, f"{task_name}.h5")
        hdf5_writer_task = {
            "result_path": result_path,
            "result_data": result_data,
            "task_name": task_name,
        }
        self._hdf5_writer = HDF5Writer(hdf5_writer_task)
        self._hdf5_writer.data_written.connect(self.on_hdf5_data_written)
        self._hdf5_writer.start()

        # create element writing task
        element_writer_task = {
            "task_path": task_path,
            "task_name": task_name,
            "resolution": self._resolution,
            "result_data": result_data,
        }
        self._element_writer = ElementWriter(element_writer_task)
        self._element_writer.data_written.connect(self.on_element_data_written)
        self._element_writer.start()

    def get_task_name(self):
        """Get the task name from the two image names."""
        left_name = self.comboBox_left.currentText()
        right_name = self.comboBox_right.currentText()
        if left_name == right_name:
            return None
        common = get_common_prefix(left_name, right_name)
        left_name = left_name.replace(common, "")
        right_name = right_name.replace(common, "")
        return f"{common}{left_name}_{right_name}"

    def restore_task(self):
        """This method restores the selected task from the task list."""
        task_name = self.comboBox_task.currentText()
        if task_name == "New" or task_name not in self._tasks:
            return
        task = self._tasks[task_name]
        self.comboBox_element.setCurrentText(task["element"])
        self.comboBox_left.setCurrentText(task["left"])
        self.comboBox_right.setCurrentText(task["right"])
        self.lineEdit_dx.setText(str(task["addX"]))
        self.lineEdit_dy.setText(str(task["addY"]))
        self.checkBox_transpose.setChecked(task["transpose"])
        self.update_left_out()

    def reset(self):
        """Reset the task."""
        self.comboBox_task.setCurrentText("New")
        self.horizontalSlider_x.setValue(50)
        self.horizontalSlider_y.setValue(50)
        self.lineEdit_dx.setText("0")
        self.lineEdit_dy.setText("0")
        self.placeholder_removed = False
        self.placeholder()


def show_splash():
    """Show the splash screen."""
    screen = (
        QDesktopWidget().screenGeometry()
    )  # Get the geometry of the screen
    width, height = screen.width(), screen.height()  # Extract width and height
    splash_size = round(min(width, height) * 0.5)
    # Create a pixmap from the main icon
    renderer = QSvgRenderer(":/icons/icons/main.svg")
    pixmap = QPixmap(splash_size, splash_size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    # Create a splash screen from the pixmap
    splash = QSplashScreen(pixmap)
    splash.show()
    return splash


def main():
    """Main function."""
    app = QApplication(sys.argv)
    splash = show_splash()
    window = MyMainWindow(splash)
    # show the main window after 3 seconds
    QTimer.singleShot(3000, window.show_maximized)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
