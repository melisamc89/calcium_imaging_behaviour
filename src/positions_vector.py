'''

Created on okt 8 Fri 2021
Author: Melisa

This script uses information that was generated by tracking_vector
which creates n_trials files for each session with the nose, ear1, ear2, head  and body positions.

Also this script will use the information from the timeline files created when analysing calcium videos
and the calcium traces to check on length of individual parts.

It needs also the information of the object locations from excels file

Will generate a discrete vector with information about the moments when the animal is in any of the objects
positional coordinates.

1. LL : lower left object
2. LR : Lower right object
3. UL : uper left object
4. UR : uper right object

'''


import os
import src.configuration
import pandas as pd
import numpy as np
import pickle
import math
import src.behavioural_analysis_functions as beh_func
from scipy.ndimage import gaussian_filter

## select mouse and session to analyze
mouse = 32363
session =2
MIN_EXPLORATION = 1 # frames taking into account videos are already in 10fps
MIN_LOOKING = 1   # frames taking into account videos are already in 10fps

RADIUS1 = 150 # in pixels, big circle
RADIUS2 = 100  # in pixels, inner circle
SPEED_LIM = 5   # speed in pixels

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
category_path = os.environ['DATA_DIR_LOCAL'] + 'ethogram/'+f'{mouse}'+'/session_'+ f'{session}'+'/'

## initial file that conteins mouse, session, trial, resting, and timestramp information.
# This table conteins all mice info
objects_file_name = current_directory + mice_directory + 'mouse_training_OS_calcium_1.xlsx'
objects_list_structure = ['condition', 'goal','group','session','drug','subject', 'trial','day', 'loc_1','loc_2']
object_list = pd.read_excel(objects_file_name)
object_list = pd.DataFrame(object_list,columns=objects_list_structure)
current_object_data = object_list[ object_list.subject ==mouse]
current_object_data = current_object_data[current_object_data.session == session]
objects = ['LR', 'LL', 'UR', 'UL']

timeline_length=[10,10,10,10,2]
session_trial = []
session_trial.append(np.arange(1,6))
session_trial.append(np.arange(6,11))
session_trial.append(np.arange(11,16))
session_trial.append(np.arange(16,21))
session_trial.append(np.arange(21,22))

day = 0
for trial_day in [1,6,11,16]:

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
    inspection_vector = np.zeros((activity.shape[1],1))
    ethogram_vector = np.zeros((activity.shape[1],1))
    ethogram_vector_ID = np.zeros((activity.shape[1],1))

    ## load tracking of behaviour
    for trial in range(len(session_trial[day])):
        ## load objects positions for this trial
        object1 = current_object_data.iloc[session_trial[day][trial]-1]['loc_1']
        object2 = current_object_data.iloc[session_trial[day][trial]-1]['loc_2']

        ## define coordinates of objects in pixels acording to the frame size
        ## and define the exploratory flag

        coordinates1 = np.array([550,100])
        exploratory_flag1= 1
        coordinates2 = np.array([150,100])
        exploratory_flag2 = 2
        coordinates3 = np.array([150,500])
        exploratory_flag3 = 3
        coordinates4 = np.array([550,500])
        exploratory_flag4 = 4

        beh_file_name = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_trial_' + \
                        f'{session_trial[day][trial]}' + '_likelihood_0.75.npy'
        beh_path = behaviour_path + beh_file_name
        if not os.path.isfile(beh_path):
            print('ERROR: Behaviour file not found:' + beh_path)
        else:
            tracking = np.load(beh_path)
            init_trial = int(timeline[trial*2])
            end_trial = int(timeline[trial*2+1])
            duration = np.min((tracking.shape[0],end_trial-init_trial))

            x_positions_pre_nose = tracking[:, 0].T
            y_positions_pre_nose = tracking[:, 1].T
            x_positions_inter_nose, y_positions_inter_nose = beh_func.interpolate_positions(x_positions_pre_nose,
                                                                                            y_positions_pre_nose)

            x_positions_nose = x_positions_inter_nose  # gaussian_filter(x_positions_inter_nose, sigma=2)
            y_positions_nose = y_positions_inter_nose  # gaussian_filter(y_positions_inter_nose, sigma=2)

            x_positions_pre_head = tracking[:, 2].T
            y_positions_pre_head = tracking[:, 3].T
            x_positions_inter_head, y_positions_inter_head = beh_func.interpolate_positions(x_positions_pre_head,
                                                                                            y_positions_pre_head)

            x_positions_head = x_positions_inter_head  # gaussian_filter(x_positions_inter_head, sigma=2)
            y_positions_head = y_positions_inter_head  # gaussian_filter(y_positions_inter_head, sigma=2)

            position = np.array([x_positions_head, y_positions_head]).T
            position_nose = np.array([x_positions_nose, y_positions_nose]).T
            vx = np.zeros((x_positions_head.shape[0], 1))
            vy = np.zeros((y_positions_head.shape[0], 1))
            vx[1:, 0] = np.diff(position[:, 0])
            vy[1:, 0] = np.diff(position[:, 1])
            speed = np.sqrt(vx * vx + vy * vy)

            ## get points coordinates for head direction and objects location
            #p2 = np.array([tracking[0:duration, 0], tracking[0:duration, 1]]).T  # nose position
            #p1 = np.array([tracking[0:duration, 2], tracking[0:duration, 3]]).T  # head position
            p2 = position_nose  # nose position
            p1 = position  # head position
            p3 = coordinates1 * np.ones_like(p1)
            p4 = coordinates2 * np.ones_like(p1)
            p5 = coordinates3 * np.ones_like(p1)
            p6 = coordinates4 * np.ones_like(p1)


            ## proximity vector between mouse position and objects
            proximity_vector1 = beh_func.proximity_vector(position, p3, radius=RADIUS1)
            proximity_vector2 = beh_func.proximity_vector(position, p4, radius=RADIUS1)
            proximity_vector3 = beh_func.proximity_vector(position, p5, radius=RADIUS1)
            proximity_vector4 = beh_func.proximity_vector(position, p6, radius=RADIUS1)

            ## select events of a certain duration

            proximity_vector1_last = beh_func.long_duration_events(proximity_vector1, MIN_EXPLORATION)
            proximity_vector2_last = beh_func.long_duration_events(proximity_vector2, MIN_EXPLORATION)
            proximity_vector3_last = beh_func.long_duration_events(proximity_vector3, MIN_EXPLORATION)
            proximity_vector4_last = beh_func.long_duration_events(proximity_vector4, MIN_EXPLORATION)

            ## now check for all data points
            for i in range(position.shape[0]):
                if position[i, 0] != -1 and position[i, 1] != -1:
                    closeness = 0
                    inspection = 0
                    if proximity_vector1_last[i]:
                            ethogram_vector[init_trial + i] = exploratory_flag1
                    else:
                        if proximity_vector2_last[i]:
                            ethogram_vector[init_trial + i] = exploratory_flag2
                        else:
                            if proximity_vector3_last[i]:
                                ethogram_vector[init_trial + i] = exploratory_flag3
                            else:
                                if proximity_vector4_last[i]:
                                    ethogram_vector[init_trial + i] = exploratory_flag4



    output_tracking_file = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_day_' + \
                        f'{day+1}' + '_likelihood_0.75_corners.npy'

    output_tracking_path = category_path + output_tracking_file
    np.save(output_tracking_path,ethogram_vector)
    day = day +1
