'''

Created on Mon 28 Sep 2020
Author: Melisa

This script will create the head direction vector

'''


import os
import src.configuration
import pandas as pd
import numpy as np
import pickle
import numpy.linalg as npalg
import datetime

## select mouse and session to analyze
mouse = 32365
session = 3
min_event_duration = 10

## source extracted calcium traces directory
calcium_directory = os.environ['PROJECT_DIR'] + 'data/calcium_activity_day_wise/'
## timeline directory
timeline_file_dir = os.environ['PROJECT_DIR'] + 'data/timeline/'
## behaviour directory
behaviour_path = os.environ['DATA_DIR_LOCAL'] + 'compiled_positions/'+f'{mouse}'+'/session_'+ f'{session}'+'/'
## output directoy
category_path = os.environ['DATA_DIR_LOCAL'] + 'head_direction/'+f'{mouse}'+'/session_'+ f'{session}'+'/'


timeline_length=[10,10,10,10,2]
session_trial = []
session_trial.append(np.arange(1,6))
session_trial.append(np.arange(6,11))
session_trial.append(np.arange(11,16))
session_trial.append(np.arange(16,21))
session_trial.append(np.arange(21,22))

day = 0
for trial_day in [1,6,11,16,21]:

    ## load calcium activity
    file_name = 'mouse_'+f'{mouse}'+'_session_'+f'{session}'+'_trial_'+f'{trial_day}' +'_v1.4.20.3.0.1.1.0.npy'
    activity = np.load(calcium_directory + file_name)

    ## load timeline
    timeline_file_path = timeline_file_dir + 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_trial_1_v' + f'{1}' + '.4.' + f'{1}' + \
                          '.' + f'{0}' + '_10.pkl'
    timeline_file= open(timeline_file_path,'rb')
    timeline_info = pickle.load(timeline_file)

    timeline = np.zeros(timeline_length[day]+1)
    for i in range(timeline_length[day]):
        timeline[i] = timeline_info[i][1]
    timeline[len(timeline)-1] = activity.shape[1]
    trial_duration = np.diff(timeline)

    ## create vector to save behaviour
    head_direction_vector = np.zeros((activity.shape[1],2))

    ## load tracking of behaviour
    for trial in range(len(session_trial[day])):

        beh_file_name = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_trial_' + \
                        f'{session_trial[day][trial]}' + '_likelihood_0.75.npy'
        beh_path = behaviour_path + beh_file_name
        if not os.path.isfile(beh_path):
            print('ERROR: File not found')
        else:
            tracking = np.load(beh_path)
            init_trial = int(timeline[trial*2])
            end_trial = int(timeline[trial*2+1])
            duration = np.min((tracking.shape[0],end_trial-init_trial))
            ## get head direction coordinates
            x_difference = tracking[0:duration,6] - tracking[0:duration,0]
            y_difference = tracking[0:duration,7] - tracking[0:duration,1]
            head_direction = np.array([x_difference , y_difference])
            head_direction = head_direction / npalg.norm(head_direction)
            head_direction_vector[init_trial:init_trial+duration,0] =head_direction[0]
            head_direction_vector[init_trial:init_trial+duration,1] =head_direction[1]


    output_tracking_file = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_day_' + \
                        f'{day+1}' + '_likelihood_0.75.npy'
    output_tracking_path = category_path + output_tracking_file
    np.save(output_tracking_path,head_direction_vector)

    day = day +1