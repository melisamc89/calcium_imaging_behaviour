'''

Created on Mon 28 Sep 2020
Author: Melisa

This script uses information that was generated by tracking_vector
which creates n_trials files for each session with the nose, ear1, ear2, head  and body positions.

Also this script will use the information from the timeline files created when analysing calcium videos
and the calcium traces to check on length of individual parts.

It needs also the information of the object locations

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
min_event_duration = 10

## object positions directory
current_directory = os.environ['PROJECT_DIR'] + 'data/scoring_sheets/'
if mouse == 32363 or mouse == 32364 or mouse == 32365 or mouse == 32366:
    mice_directory = '32363-32366/'
else:
    mice_directory = '56165-56166/'

## source extracted calcium traces directory
calcium_directory = os.environ['PROJECT_DIR'] + 'data/calcium_activity_day_wise/'
## timeline directory
timeline_file_dir = os.environ['PROJECT_DIR'] + 'data/timeline/'
## behaviour directory
behaviour_path = os.environ['DATA_DIR_LOCAL'] + 'compiled_positions/'+f'{mouse}'+'/session_'+ f'{session}'+'/'
## output directoy
category_path = os.environ['DATA_DIR_LOCAL'] + 'category_behaviours/'+f'{mouse}'+'/session_'+ f'{session}'+'/'

## initial file that conteins mouse, session, trial, resting, and timestramp information.
# This table conteins all mice info
objects_file_name = current_directory + mice_directory + 'mouse_training_OS_calcium_1.xlsx'
objects_list_structure = ['condition', 'goal','group','session','drug','subject', 'trial','day', 'loc_1','loc_2']
object_list = pd.read_excel(objects_file_name)
object_list = pd.DataFrame(object_list,columns=objects_list_structure)
current_object_data = object_list[ object_list.subject ==mouse]
current_object_data = current_object_data[current_object_data.session == session]
objects = ['LR', 'LL', 'UR', 'UL']

## define radious of circle to define behaviour
EXPLORING_RADIOUS = 100 #pixels

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
    behaviour_vector = np.zeros((activity.shape[1],1))

    ## load tracking of behaviour
    for trial in range(len(session_trial[day])):
        ## load objects positions for this trial
        object1 = current_object_data.iloc[session_trial[day][trial]-1]['loc_1']
        object2 = current_object_data.iloc[session_trial[day][trial]-1]['loc_2']

        ## define coordinates of objects in pixels acording to the frame size
        ## and define the exploratory flag

        if object1 == 'LL':
            coordinates1 = np.array([650,600])
            exploratory_flag1 = 2
        if object1 == 'LR':
            coordinates1 = np.array([225,600])
            exploratory_flag1 = 3
        if object1 == 'UR':
            coordinates1 = np.array([225,200])
            exploratory_flag1 = 4
        if object1 == 'UL':
            coordinates1 = np.array([650,200])
            exploratory_flag1 = 5

        if object2 == 'LL':
            coordinates2 = np.array([650,600])
            exploratory_flag2 = 2
        if object2 == 'LR':
            coordinates2 = np.array([225,600])
            exploratory_flag2 = 3
        if object2 == 'UR':
            coordinates2 = np.array([225,200])
            exploratory_flag2 = 4
        if object2 == 'UL':
            coordinates2 = np.array([650,200])
            exploratory_flag2 = 5

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
            ## get tracking
            x_positions = np.mean(tracking[0:2:duration,[0,2,4,6,8]],axis = 1)
            y_positions = np.mean(tracking[0:2:duration,[1,3,5,7,9]],axis = 1)
            vector_position = np.array([x_positions, y_positions]).T

            for i in range(vector_position.shape[0]):
                distance1 = npalg.norm(vector_position[i] - coordinates1)
                distance2 = npalg.norm(vector_position[i] - coordinates2)
                if distance1 > EXPLORING_RADIOUS and distance2 > EXPLORING_RADIOUS:
                    behaviour_vector[init_trial+i] = 1
                else:
                    if distance1 < EXPLORING_RADIOUS:
                        behaviour_vector[init_trial + i] = exploratory_flag1
                    else:
                        if distance2 < EXPLORING_RADIOUS:
                            behaviour_vector[init_trial + i] = exploratory_flag2

    output_tracking_file = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_day_' + \
                        f'{trial_day}' + '_likelihood_0.75.npy'
    output_tracking_path = category_path + output_tracking_file
    np.save(output_tracking_path,behaviour_vector)

    day = day +1