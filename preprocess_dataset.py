'''
 Module for preprocessing the CK+48 dataset:
    1. Load dataset
    2. Shuffle dataset
    3. Split into test and trainning datasets
'''


import numba as np
import cv2 as cv2


def load_dataset():
    """
    A function that loads all data images from the CK+48 dataset folder

    Args:
        param1: input image name.
        param2: sigma value for the window function w(x, y). 
                (default is 2)
        param3: threshold for min(λ1, λ2) to detect corners.
                (default 0: we don't detect corners and plot result image, 
                just compute the eigenvalues; if the function takes an input
                for parameter threshold, then we use that for detecing corners
                and plot result corresponding image)
    Returns:
        output1: a list of λ1s.
        output2: a list of λ2s.
    """