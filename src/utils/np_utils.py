import os
import numpy as np


def save_np_arrays(results_dir, dict_array):
    os.makedirs(results_dir, exist_ok=True)
    for fname, array in dict_array.items():
        np.save(os.path.join(results_dir, fname), array)
