'''

Created on Wed 23 Sep 2020
Author: Melisa

This script contains the steps into opening and construction the analysis for the
fly camera from the object space task.

This will plot a few examples of the tracking for some body parts, using a particular threshold
for the likelihood of the tracking

'''

import os
import configuration
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import math


## select mouse and session to analyze
mouse = 56166
session = 2

## name data excel file where list of file names is located
file_names_file = os.environ['PROJECT_DIR_LOCAL'] + 'calcium_imaging_paths_behaviour.xlsx'

## load excel file and convert to data frame
file_names_excel= pd.read_excel(file_names_file)
file_names_structure = ['index','mouse','session','trial','is_rest','experimenter','date',
                        'time','raw_data_behaviour','raw_file']
file_names_df= pd.DataFrame(file_names_excel ,columns=file_names_structure )

## select the specific mouse and session
mouse_selection = file_names_df[file_names_df.mouse == mouse]
session_selection = mouse_selection[mouse_selection.session == session]

## define individual inputs
#dlc_extension = 'DLC_resnet50_object_spacesep21shuffle1_50000.csv'
dlc_extension = 'DLC_resnet50_object_spaceapr12shuffle1_50000.csv'
input_path = os.environ['DATA_DIR_LOCAL'] + 'dlc_2021/'+f'{mouse}'+'/session_'+ f'{session}'+'/'
output_path = os.environ['DATA_DIR_LOCAL'] + 'compiled_positions/'+f'{mouse}'+'/session_'+ f'{session}'+'/'
## define the structure of simgle data files
#body_parts = ['nose', 'ear1', 'ear2', 'head', 'middle_body', 'tail_start', 'tail_middle', 'tail_end']
body_parts = ['nose', 'head', 'ear1', 'ear2', 'middle_body', 'start_tail']
body_part_structure = ['x', 'y', 'likelihood']

## define likelihood for data selection
LIKELIHOOD = 0.75
sf =20
re_sf = 2

for trial in range(len(session_selection)):
    ##define path and load data from tracking of one trial
    if type(session_selection.iloc[trial]['raw_file'])==str:
        input_file_path = input_path + session_selection.iloc[trial]['raw_file'][:-4]+dlc_extension
        output_file_path = output_path + 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_trial_' + f'{trial+1}' +'_likelihood_'+ f'{LIKELIHOOD}'+ '.npy'
        tracking_data = pd.read_csv(input_file_path)
        tracking_data_array = tracking_data.to_numpy()

        ## select: nose, ear1, ear2, head and middle body positions for the tracking
        x_nose = np.round(tracking_data_array[2:,1].astype(np.float),2)
        y_nose = np.round(tracking_data_array[2:,2].astype(np.float),2)
        likelihood_nose = np.round(tracking_data_array[2:,3].astype(np.float),2)

        x_ear1 = np.round(tracking_data_array[2:,7].astype(np.float),2)
        y_ear1 = np.round(tracking_data_array[2:,8].astype(np.float),2)
        likelihood_ear1 = np.round(tracking_data_array[2:,6].astype(np.float),2)

        x_ear2 = np.round(tracking_data_array[2:,10].astype(np.float),2)
        y_ear2 = np.round(tracking_data_array[2:,11].astype(np.float),2)
        likelihood_ear2 = np.round(tracking_data_array[2:,9].astype(np.float),2)

        x_head = np.round(tracking_data_array[2:,4].astype(np.float),2)
        y_head = np.round(tracking_data_array[2:,5].astype(np.float),2)
        likelihood_head = np.round(tracking_data_array[2:,12].astype(np.float),2)

        x_body = np.round(tracking_data_array[2:, 13].astype(np.float), 2)
        y_body = np.round(tracking_data_array[2:, 14].astype(np.float), 2)
        likelihood_body = np.round(tracking_data_array[2:, 15].astype(np.float), 2)

        selection_nose = np.where(likelihood_nose>LIKELIHOOD)
        selection_ear1 = np.where(likelihood_ear1>LIKELIHOOD)
        selection_ear2 = np.where(likelihood_ear2>LIKELIHOOD)
        selection_head = np.where(likelihood_head>LIKELIHOOD)
        selection_body = np.where(likelihood_body>LIKELIHOOD)

        intersec1 = np.intersect1d(selection_nose,selection_ear1)
        intersec2 = np.intersect1d(selection_ear2,selection_head)
        intersec3 = np.intersect1d(intersec1, selection_body)
        selection= np.intersect1d(intersec2,intersec3)

        new_x_nose = (-1000)*np.ones_like(x_nose)
        new_y_nose = (-1000)*np.ones_like(y_nose)
        new_x_nose[selection] = x_nose[selection]
        new_y_nose[selection] = y_nose[selection]

        new_x_ear1 = (-1000)*np.ones_like(x_ear1)
        new_y_ear1 = (-1000)*np.ones_like(y_ear1)
        new_x_ear1[selection] = x_ear1[selection]
        new_y_ear1[selection] = y_ear1[selection]

        new_x_ear2 = (-1000)*np.ones_like(x_ear2)
        new_y_ear2 = (-1000)*np.ones_like(y_ear2)
        new_x_ear2[selection] = x_ear2[selection]
        new_y_ear2[selection] = y_ear2[selection]

        new_x_head = (-1000)*np.ones_like(x_head)
        new_y_head = (-1000)*np.ones_like(y_head)
        new_x_head[selection] = x_head[selection]
        new_y_head[selection] = y_head[selection]

        new_x_body = (-1000)*np.ones_like(x_body)
        new_y_body = (-1000)*np.ones_like(y_body)
        new_x_body[selection] = x_body[selection]
        new_y_body[selection] = y_body[selection]

        #new_tracking = np.array([new_x_nose, new_y_nose, new_x_ear1, new_y_ear1, new_x_ear2, new_y_ear2, new_x_head, new_y_head, new_x_body, new_y_body])
        #new_tracking = np.array(new_tracking).T
        new_tracking = np.array([new_x_nose, new_y_nose, new_x_head, new_y_head, new_x_ear1, new_y_ear1, new_x_ear2, new_y_ear2, new_x_body, new_y_body])


        reshape_tracking = np.reshape(new_tracking[:, :int(int(new_tracking.shape[1] / re_sf) * re_sf)],
                                             (new_tracking.shape[0], int(new_tracking.shape[1] / re_sf), re_sf))
        resample_tracking_mean = np.mean(reshape_tracking, axis=2)

        np.save(output_file_path,resample_tracking_mean.T)


