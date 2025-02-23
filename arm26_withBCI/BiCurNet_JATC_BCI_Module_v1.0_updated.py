# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 17:26:21 2023

@author: INFINITE-WORKSTATION
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation
from scipy.signal.windows import gaussian, hamming
from scipy.ndimage import filters

xval = np.load('BiCurNet_anant_240_1600_xval.npy')

yval = np.load('BiCurNet_anant_240_1600_yval.npy')
yval_flatten = np.load('BiCurNet_anant_240_1600_yval_flatten.npy')
pcc = np.load('BiCurNet_anant_240_1600_pcc.npy')

BiCurNet_trained = tf.keras.models.load_model('BiCurNet_anant_240_1600')

# Show the model architecture
BiCurNet_trained.summary()

filt_weight = gaussian(10, 8)

def rescale_array(arr, old_min, old_max, new_min, new_max):
    # Check if any value in the array is outside the old range
    if np.any((arr < old_min) | (arr > old_max)):
        raise ValueError("Input values are not within the old range")

    # Calculate the ratios for rescaling
    ratios = (arr - old_min) / (old_max - old_min)

    # Map the ratios to the new range to get the rescaled array
    rescaled_array = new_min + ratios * (new_max - new_min)
    return rescaled_array

# Define the old range and new range
old_min = 0.00
old_max = 0.12
new_min = 0.0
new_max = 0.9

for frame in range(yval.shape[0]):
    a=xval[frame]
    b=a.reshape(1,a.shape[0],a.shape[1])
    # Evaluate the restored model
    output = BiCurNet_trained.predict(b)
    updated_angle = output.transpose()
    filt_angle = filters.convolve1d(updated_angle, filt_weight / filt_weight.sum())
    # scaled_angle = rescale_array(filt_angle, old_min, old_max, new_min, new_max)
    
    time.sleep(1.5)