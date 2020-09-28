'''

Created on Mon 28 Sep 2020
Author: Melisa
This script will create a video with the mouse moving, the mean positions and head direction of the animal.
'''


import os
import src.configuration
import pandas as pd
import numpy as np
import pickle
import numpy.linalg as npalg
import datetime

## select mouse and session to analyze
mouse = 32363
session = 1
trial = 1

## behaviour directory
behaviour_path = os.environ['DATA_DIR_LOCAL'] + 'compiled_positions/'+f'{mouse}'+'/session_'+ f'{session}'+'/'
## output directoy
video_path = os.environ['DATA_DIR_LOCAL'] + 'head_direction_video/'+f'{mouse}'+'/session_'+ f'{session}'+'/'


beh_file_name = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_trial_' + \
                f'{trial}' + '_likelihood_0.75.npy'

beh_path = behaviour_path + beh_file_name
tracking = np.load(beh_path)

## tracking coordinates
x_positions = np.mean(tracking[:, [0, 2, 4, 6, 8]], axis=1).T
y_positions = np.mean(tracking[:, [1, 3, 5, 7, 9]], axis=1).T

## get head direction coordinates
x_difference = (tracking[:,0] - tracking[:,6]).T
y_difference = (tracking[:,1] - tracking[:,7]).T
head_direction = np.array([x_difference , y_difference])
head_direction = head_direction *10 / npalg.norm(head_direction)



