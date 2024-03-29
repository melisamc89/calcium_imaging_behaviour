'''

Created on Wed 07 Oct 2020
Author: Melisa

This script uses information that was generated by tracking_vector
which creates n_trials files for each session with the nose, ear1, ear2, head  and body positions.

Also this script will use the information from the timeline files created when analysing calcium videos
and the calcium traces to check on length of individual parts.

It needs also the information of the object locations from excels file

Will generate a discrete vector with information about the moments when the animal is involved in a certaing
exploratory behaviour

'''


import os
import src.configuration
import pandas as pd
import numpy as np
import pickle
import math
import src.behavioural_analysis_functions as beh_func

## select mouse and session to analyze
mouse = 32363
session = 1
MIN_EXPLORATION = 10 # frames taking into account videos are already in 10fps
MIN_LOOKING = 20    # frames taking into account videos are already in 10fps

RADIUS1 = 200 # in pixels, big circle
RADIUS2 = 75  # in pixels, inner circle

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
    inspection_vector = np.zeros((activity.shape[1],1))

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
            looking_flag1 = 2
        if object1 == 'LR':
            coordinates1 = np.array([225,600])
            exploratory_flag1 = 3
            looking_flag1 = 3
        if object1 == 'UR':
            coordinates1 = np.array([225,200])
            exploratory_flag1 = 4
            looking_flag1 = 4
        if object1 == 'UL':
            coordinates1 = np.array([650,200])
            exploratory_flag1 = 5
            looking_flag1 = 5

        if object2 == 'LL':
            coordinates2 = np.array([650,600])
            exploratory_flag2 = 2
            looking_flag2 = 2
        if object2 == 'LR':
            coordinates2 = np.array([225,600])
            exploratory_flag2 = 3
            looking_flag2 = 3
        if object2 == 'UR':
            coordinates2 = np.array([225,200])
            exploratory_flag2 = 4
            looking_flag2 = 4
        if object2 == 'UL':
            coordinates2 = np.array([650,200])
            exploratory_flag2 = 5
            looking_flag2 = 5

        beh_file_name = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_trial_' + \
                        f'{session_trial[day][trial]}' + '_likelihood_0.75.npy'
        beh_path = behaviour_path + beh_file_name
        if not os.path.isfile(beh_path):
            print('ERROR: File not found')
        else:
            tracking = np.load(beh_path)
            #tracking_load = np.load(beh_path).T
            #tracking_reshape = np.reshape(tracking_load[:, :int(int(tracking_load.shape[1] / 2) * 2)],
            #    (tracking_load.shape[0], int(tracking_load.shape[1] / 2), 2))
            #tracking_resample = np.mean(tracking_reshape, axis=2)
            #tracking = tracking_resample.T

            init_trial = int(timeline[trial*2])
            end_trial = int(timeline[trial*2+1])
            duration = np.min((tracking.shape[0],end_trial-init_trial))
            ## tracking coordinates
            x_positions = np.mean(tracking[0:duration, [0, 2, 4, 6, 8]], axis=1).T
            y_positions = np.mean(tracking[0:duration, [1, 3, 5, 7, 9]], axis=1).T
            position = np.array([x_positions, y_positions]).T

            ## get points coordinates for head direction and objects location
            p2 = np.array([tracking[0:duration, 0], tracking[0:duration, 1]]).T  # nose position
            p1 = np.array([tracking[0:duration, 6], tracking[0:duration, 7]]).T  # head position
            p3 = coordinates1 * np.ones_like(p1)
            p4 = coordinates2 * np.ones_like(p1)

            ## binary looking at object vectors
            looking_vector1, angle1_vector = beh_func.looking_at_vector(p2, p1, p3)
            looking_vector2, angle2_vector = beh_func.looking_at_vector(p2, p1, p4)

            ## proximity vector between mouse position and objects
            proximity_vector1 = beh_func.proximity_vector(position, p3, radius=RADIUS1)
            proximity_vector2 = beh_func.proximity_vector(position, p4, radius=RADIUS1)

            ## super proximity vector for mouse position and objects (closer that proximity1)
            super_proximity_vector1 = beh_func.proximity_vector(position, p3, radius=RADIUS2)
            super_proximity_vector2 = beh_func.proximity_vector(position, p4, radius=RADIUS2)

            ## select events of a certain duration
            looking_vector1_last = beh_func.long_duration_events(looking_vector1, MIN_LOOKING)
            looking_vector2_last = beh_func.long_duration_events(looking_vector2, MIN_LOOKING)

            proximity_vector1_last = beh_func.long_duration_events(proximity_vector1, MIN_EXPLORATION*2)
            proximity_vector2_last = beh_func.long_duration_events(proximity_vector2, MIN_EXPLORATION*2)

            super_proximity_vector1_last = beh_func.long_duration_events(super_proximity_vector1, MIN_EXPLORATION)
            super_proximity_vector2_last = beh_func.long_duration_events(super_proximity_vector2, MIN_EXPLORATION)

            ## now check for all data points
            for i in range(position.shape[0]):
                if position[i, 0] != 0 and position[i, 1] != 0:
                    if proximity_vector1_last[i] and not math.isnan(angle1_vector[i]):
                        if looking_vector1_last[i] or super_proximity_vector1_last[i]:
                            behaviour_vector[init_trial + i] = exploratory_flag1
                    else:
                        if proximity_vector2_last[i] and not math.isnan(angle2_vector[i]):
                            if looking_vector2[i] or super_proximity_vector1[i]:
                                behaviour_vector[init_trial + i] = exploratory_flag2
                        else:
                            behaviour_vector[init_trial + i] = 1
                    if looking_vector1_last[i] and not looking_vector2_last[i]:
                        inspection_vector[init_trial + i] = looking_flag1
                    else:
                        if looking_vector2_last[i] and not looking_vector1_last[i]:
                            inspection_vector[init_trial + i] = looking_flag2
                        else:
                            inspection_vector[init_trial + i ] = 1


    output_tracking_file = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_day_' + \
                        f'{day+1}' + '_likelihood_0.75.npy'
    output_tracking_path = category_path + output_tracking_file
    np.save(output_tracking_path,behaviour_vector)

    output_inspection_file = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_day_' + \
                        f'{day+1}' + '_likelihood_0.75_inspection.npy'
    output_inspection_path = category_path + output_inspection_file
    np.save(output_inspection_path,inspection_vector)

    day = day +1

