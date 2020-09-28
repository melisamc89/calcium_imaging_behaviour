'''

Created on Fri 25 Sep 2020
Author: Melisa

This script uses information that was generated by compile_positions.py
which creates n_trials files for each session with the nose, ear1, ear2, head and body positions.

Also this script will use the information from the timeline files created when analysing calcium videos
and the calcium traces to check on length of individual parts.

'''


import os
import src.configuration
import pandas as pd
import numpy as np
import pickle
import datetime

## select mouse and session to analyze
mouse = 32363
session = 1
min_event_duration = 10

## source extracted calcium traces directory
calcium_directory = os.environ['PROJECT_DIR'] + 'data/calcium_activity_day_wise/'
## timeline directory
timeline_file_dir = os.environ['PROJECT_DIR'] + 'data/timeline/'
## behaviour directory
behaviour_path = os.environ['DATA_DIR_LOCAL'] + 'compiled_positions/'+f'{mouse}'+'/session_'+ f'{session}'+'/'
## output directoy
tracking_path = os.environ['DATA_DIR_LOCAL'] + 'tracking/'+f'{mouse}'+'/session_'+ f'{session}'+'/'


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
    tracking_vector = (-1)*np.ones((activity.shape[1],2))

    ## load tracking of behaviour
    for trial in range(len(session_trial[day])):
        beh_file_name = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_trial_' + \
                        f'{session_trial[day][trial]}' + '_likelihood_0.75.npy'
        beh_path = behaviour_path + beh_file_name
        tracking = np.load(beh_path)
        init_trial = int(timeline[trial*2])
        end_trial = int(timeline[trial*2+1])
        duration = np.min((tracking.shape[0],end_trial-init_trial))
        x_positions = np.mean(tracking[0:duration,[0,2,4,6,8]],axis = 1)
        y_positions = np.mean(tracking[0:duration,[1,3,5,7,9]],axis = 1)
        tracking_vector[init_trial:init_trial+duration,0] =x_positions
        tracking_vector[init_trial:init_trial+duration,1] =y_positions

    output_tracking_file = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_day_' + \
                        f'{day+1}' + '_likelihood_0.75.npy'
    output_tracking_path = tracking_path + output_tracking_file
    np.save(output_tracking_path,tracking_vector)

    day = day +1