import os

import numpy as np
import pandas as pd
from PIL import Image


def read_element_data(folder_full_path):
    """Read element data from a folder."""
    included_elements = []
    file_list = [
        f
        for f in os.listdir(folder_full_path)
        if f.endswith(".csv") and "Point" not in f
    ]

    if not file_list:
        raise ValueError(f"Folder {folder_full_path} is empty")

    temp_file_path = os.path.join(folder_full_path, file_list[0])
    total_counts = data_read(temp_file_path)
    resolution = obtain_resolution(temp_file_path)

    element_data_container = {}
    for file in file_list[1:]:
        temp_file_path = os.path.join(folder_full_path, file)
        temp_data = data_read(temp_file_path)

        total_counts += temp_data
        temp_element_name = file.split("_")[-1].split(".")[0].split(" ")[0]

        if temp_element_name != "Grey":
            included_elements.append(temp_element_name)

        temp_data = remove_noise_data(temp_data)
        element_data_container[temp_element_name] = temp_data

    if "Grey" not in element_data_container:
        file_list = [
            f
            for f in os.listdir(folder_full_path)
            if f.endswith(".tif") and "Grey" in f
        ]
        if file_list:
            grey_image = Image.open(
                os.path.join(folder_full_path, file_list[0])
            )
            element_data_container["Grey"] = np.array(
                grey_image, dtype=np.uint8
            )

    if "Grey" in element_data_container:
        element_data_container["Grey"] = np.array(
            element_data_container["Grey"], dtype=np.uint8
        )

    return (
        element_data_container["Grey"],
        included_elements,
        element_data_container,
        resolution,
        total_counts,
    )


def read_element_csv(file_path):
    """Read element data from a csv file."""
    temp_data = pd.read_csv(file_path, skiprows=5, header=None).to_numpy()
    return temp_data


def obtain_resolution(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        for _ in range(4):
            temp_text = file.readline()
        if "um" in temp_text:
            resolution = float(temp_text.split("Size,")[1].split("um")[0])
        else:
            resolution = (
                float(temp_text.split("Size,")[1].split("nm")[0]) / 1000
            )  # Convert nm to um
    return resolution


