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
