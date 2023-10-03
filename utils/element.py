import os

import numpy as np
import pandas as pd
from PIL import Image



def data_read(file_path):
    if os.name == "posix":
        temp_data = pd.read_csv(file_path, skiprows=5).to_numpy()
    else:
        temp_data = pd.read_excel(file_path, skiprows=5).to_numpy()
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


