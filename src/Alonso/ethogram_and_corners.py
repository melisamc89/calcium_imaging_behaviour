'''

Created on Thrus 31 Mar 2022
Author: Melisa


'''

import os
import src.Alonso.configuration
import pandas as pd
import numpy as np
import pickle
import math
import src.behavioural_analysis_functions as beh_func
from scipy.ndimage import gaussian_filter
import scipy.io as sciio
import numpy.linalg as npalg
from scipy import signal


def head_direction(pos):
    # get head direction coordinates
    x_diff = pos[1,0] - pos[0,0]
    y_diff = pos[1,1] - pos[0,1]
    hd = np.array([x_diff , y_diff])
    norm = np.sqrt(x_diff*x_diff + y_diff*y_diff)
    if norm:
        hd = hd / np.sqrt(x_diff*x_diff + y_diff*y_diff)
    else:
        hd = [0,0]
    return hd

def body_direction(pos):
    # get body direction coordinates
    x_diff = pos[0,2] - pos[0,1]
    y_diff = pos[1,2] - pos[1,1]
    bd = np.array([x_diff , y_diff])
    #bd = bd / npalg.norm(bd)
    return bd

def distance(point1,point2):
    xdiff = point1[0] - point2[0]
    ydiff = point1[1] - point2[1]
    dist = np.sqrt(xdiff*xdiff + ydiff*ydiff)
    return dist

## define trials for each day
timeline_length=[10,10,10,10,2]
session_trial = []
session_trial.append(np.arange(1,6))
session_trial.append(np.arange(6,11))
session_trial.append(np.arange(11,16))
session_trial.append(np.arange(16,21))
session_trial.append(np.arange(21,22))

def objects_position(current_object_data = None, day = None, trial = None, coordinates = None):
    ## load objects positions for this trial
    object1 = current_object_data.iloc[session_trial[day][trial]-1]['loc_1']
    object2 = current_object_data.iloc[session_trial[day][trial]-1]['loc_2']
    ## define coordinates of objects in pixels acording to the frame size

    objects = ['LR', 'LL', 'UR', 'UL']

    selected_1 = -1
    selected_2 = -1
    for i in range(4):
        for j in range(4):
            if object1 == objects[i]:
                coordinates1 = coordinates[i]
                exploratory_flag1 = i + 3
                looking_flag1 = i + 7
                selected_1 = i+1
            if object2 == objects[j]:
                coordinates2 = coordinates[j]
                exploratory_flag2 = j + 3
                looking_flag2 = j + 7
                selected_2 = j+1

    return [coordinates1, exploratory_flag1, selected_1, looking_flag1, coordinates2, exploratory_flag2, selected_2, looking_flag2]

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    if np.isnan(angle):
        angle = 0
    return angle


## select mouse and session to analyze
mouse = 401714
MIN_EXPLORATION = 1 # frames taking into account videos are already in 10fps
MIN_LOOKING = 1   # frames taking into account videos are already in 10fps

RADIUS1 = 150 # in pixels, big circle
RADIUS2 = 100  # in pixels, inner circle
SPEED_LIM = 5   # speed in pixels

## object positions directory
## object positions directory
current_directory = os.environ['DATA_DIR_LOCAL'] + f'{mouse}' + '/'

# This table conteins all mice info
objects_file_name = current_directory + 'training_sheet_401714.xlsx'
objects_list_structure = ['condition', 'goal', 'group', 'session', 'drug', 'subject', 'trial', 'day', 'loc_1', 'loc_2']

object_list = pd.read_excel(objects_file_name)
object_list = pd.DataFrame(object_list, columns=objects_list_structure)

coordinates = [np.array([800, 250]), np.array([350, 200]), np.array([350, 700]), np.array([800, 700])]
objects = ['LR', 'LL', 'UR', 'UL']

for session in [1,2]:

    ## source extracted calcium traces directory
    calcium_directory = os.environ['DATA_DIR_LOCAL'] +f'{mouse}'+'/' + 'data/calcium_activity_day_wise/'
    ## timeline directory
    timeline_file_dir =os.environ['DATA_DIR_LOCAL'] +f'{mouse}'+'/' + 'data/timeline/'

    ## behaviour directory
    ## behaviour directory
    behaviour_path = os.environ['PROJECT_DIR_LOCAL'] + 'data/compiled_positions/'+f'{mouse}'+'/week'+ f'{session}'+'/'
    ## output directoy
    category_path = os.environ['PROJECT_DIR_LOCAL']  + 'data/ethogram/'+f'{mouse}'+'/week'+ f'{session}'+'/'

    ## initial file that conteins mouse, session, trial, resting, and timestramp information.

    # current_object_data = object_list[ object_list.subject == mouse]
    current_object_data = object_list[object_list.session == session]

    timeline_length=[10,10,10,10,2]
    session_trial = []
    session_trial.append(np.arange(1,6))
    session_trial.append(np.arange(6,11))
    session_trial.append(np.arange(11,16))
    session_trial.append(np.arange(16,21))
    session_trial.append(np.arange(21,22))

    day = 0
    for trial_day in [1,6,11,16]:
        print(day)
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
        ethogram_vector = np.zeros((activity.shape[1], 1))
        corners_vector = np.zeros((activity.shape[1], 1))
        speed_vector = np.zeros((activity.shape[1], 1))

        ## load tracking of behaviour
        for trial in range(len(session_trial[day])):
            ## load objects positions for this trial
            ## define coordinates of objects in pixels acording to the frame size
            ## and define the exploratory flag

            [coordinates1,exploratory_flag1, selected_1, looking_flag1,
            coordinates2,exploratory_flag2,selected_2, looking_flag2] = \
                objects_position(current_object_data = current_object_data, day = day, trial = trial, coordinates= coordinates)

            objects = [selected_1, selected_2]
            corners_list = [1, 2, 3, 4]
            non_object_list = list(set(objects) ^ set(corners_list))

            beh_file_name = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_trial_' + \
                            f'{session_trial[day][trial]}' + '_likelihood_0.75.npy'
            beh_path = behaviour_path + beh_file_name
            if not os.path.isfile(beh_path):
                print('ERROR: Behaviour file not found:' + beh_path)
            else:
                tracking = np.load(beh_path)

                init_trial = int(timeline[trial * 2])
                end_trial = int(timeline[trial * 2 + 1])
                duration = np.min((tracking.shape[0], end_trial - init_trial))
                ## tracking coordinates
                cm = np.zeros((tracking.shape[0], 2))
                for body_part in range(5):
                    x = tracking[:, body_part * 2]
                    y = tracking[:, body_part * 2 + 1]
                    x_new, y_new = beh_func.interpolate_positions(x, y)
                    cm[:, 0] = signal.medfilt(x_new, 7)
                    cm[:, 1] = signal.medfilt(y_new, 7)

                x_positions_pre_nose = tracking[:, 0].T
                y_positions_pre_nose = tracking[:, 1].T
                x_positions_inter_nose, y_positions_inter_nose = beh_func.interpolate_positions(
                    x_positions_pre_nose,
                    y_positions_pre_nose)
                x_positions_nose = x_positions_inter_nose  # gaussian_filter(x_positions_inter_nose, sigma=2)
                y_positions_nose = y_positions_inter_nose  # gaussian_filter(y_positions_inter_nose, sigma=2)

                x_positions_pre_head = tracking[:, 2].T
                y_positions_pre_head = tracking[:, 3].T
                x_positions_inter_head, y_positions_inter_head = beh_func.interpolate_positions(
                    x_positions_pre_head,
                    y_positions_pre_head)
                x_positions_head = x_positions_inter_head  # gaussian_filter(x_positions_inter_head, sigma=2)
                y_positions_head = y_positions_inter_head  # gaussian_filter(y_positions_inter_head, sigma=2)

                position = np.array([cm[:, 0], cm[:, 1]]).T
                position_nose = np.array([x_positions_nose, y_positions_nose]).T
                vx = np.zeros((cm[:, 0].shape[0], 1))
                vy = np.zeros((cm[:, 1].shape[0], 1))
                vx[1:, 0] = np.diff(position[:, 0])
                vy[1:, 0] = np.diff(position[:, 1])
                speed = np.sqrt(vx * vx + vy * vy)

                ## get points coordinates for head direction and objects location
                # p2 = np.array([tracking[0:duration, 0], tracking[0:duration, 1]]).T  # nose position
                # p1 = np.array([tracking[0:duration, 2], tracking[0:duration, 3]]).T  # head position
                p2 = position_nose  # nose position
                p0 = np.array([x_positions_head, y_positions_head]).T
                p1 = position  # cm position
                p3 = coordinates1 * np.ones_like(p1)
                p4 = coordinates2 * np.ones_like(p1)
                p5 = coordinates[non_object_list[0]-1] * np.ones_like(p1)
                p6 = coordinates[non_object_list[1]-1] * np.ones_like(p1)

                ## binary looking at object vectors
                looking_vector1, angle1_vector = beh_func.looking_at_vector(p2, p0, p3)
                looking_vector2, angle2_vector = beh_func.looking_at_vector(p2, p0, p4)
                looking_vector3, angle3_vector = beh_func.looking_at_vector(p2, p0, p5)
                looking_vector4, angle4_vector = beh_func.looking_at_vector(p2, p0, p6)

                ## proximity vector between mouse position and objects
                proximity_vector1 = beh_func.proximity_vector(position, p3, radius=RADIUS1)
                proximity_vector2 = beh_func.proximity_vector(position, p4, radius=RADIUS1)
                proximity_vector3 = beh_func.proximity_vector(position, p5, radius=RADIUS1)
                proximity_vector4 = beh_func.proximity_vector(position, p6, radius=RADIUS1)

                ## super proximity vector for mouse position and objects (closer that proximity1)
                super_proximity_vector1 = beh_func.proximity_vector(position, p3, radius=RADIUS2)
                super_proximity_vector2 = beh_func.proximity_vector(position, p4, radius=RADIUS2)
                super_proximity_vector3 = beh_func.proximity_vector(position, p5, radius=RADIUS2)
                super_proximity_vector4 = beh_func.proximity_vector(position, p6, radius=RADIUS2)

                ## select events of a certain duration
                looking_vector1_last = beh_func.long_duration_events(looking_vector1, MIN_LOOKING)
                looking_vector2_last = beh_func.long_duration_events(looking_vector2, MIN_LOOKING)
                looking_vector3_last = beh_func.long_duration_events(looking_vector3, MIN_LOOKING)
                looking_vector4_last = beh_func.long_duration_events(looking_vector4, MIN_LOOKING)

                ##
                proximity_vector1_last = beh_func.long_duration_events(proximity_vector1, MIN_EXPLORATION)
                proximity_vector2_last = beh_func.long_duration_events(proximity_vector2, MIN_EXPLORATION)
                proximity_vector3_last = beh_func.long_duration_events(proximity_vector3, MIN_EXPLORATION)
                proximity_vector4_last = beh_func.long_duration_events(proximity_vector4, MIN_EXPLORATION)

                super_proximity_vector1_last = beh_func.long_duration_events(super_proximity_vector1,
                                                                                 MIN_EXPLORATION)
                super_proximity_vector2_last = beh_func.long_duration_events(super_proximity_vector2,
                                                                                 MIN_EXPLORATION)
                super_proximity_vector3_last = beh_func.long_duration_events(super_proximity_vector3, MIN_EXPLORATION)
                super_proximity_vector4_last = beh_func.long_duration_events(super_proximity_vector4, MIN_EXPLORATION)

                ## now check for all data points
                for i in range(position.shape[0]):
                    speed_vector[init_trial + i] = speed[i]
                    if position[i, 0] > 0 and position[i, 1] > 0:
                        closeness = 0
                        inspection = 0
                        if proximity_vector1_last[i] and not math.isnan(angle1_vector[i]):
                            if looking_vector1_last[i] or super_proximity_vector1_last[i]:
                                ethogram_vector[init_trial + i] = exploratory_flag1
                                corners_vector[init_trial + i] = corners_list[selected_1 - 1]
                                closeness = 1
                        else:
                            if proximity_vector2_last[i] and not math.isnan(angle2_vector[i]):
                                if looking_vector2_last[i] or super_proximity_vector2_last[i]:
                                    ethogram_vector[init_trial + i] = exploratory_flag2
                                    corners_vector[init_trial + i] = corners_list[selected_2 - 1]
                                    closeness = 1
                            else:
                                if proximity_vector3_last[i] and not math.isnan(angle3_vector[i]):
                                    if looking_vector3_last[i] or super_proximity_vector3_last[i]:
                                        corners_vector[init_trial + i] = corners_list[non_object_list[0] - 1]
                                else:
                                    if proximity_vector4_last[i] and not math.isnan(angle4_vector[i]):
                                        if looking_vector4_last[i] or super_proximity_vector4_last[i]:
                                            corners_vector[init_trial + i] = corners_list[non_object_list[1] - 1]
                            if speed[i] > SPEED_LIM and closeness == 0:
                                if looking_vector1_last[i] and not looking_vector2_last[i]:
                                    ethogram_vector[init_trial + i] = looking_flag1
                                    inspection = 1
                                if looking_vector2_last[i] and not looking_vector1_last[i]:
                                    ethogram_vector[init_trial + i] = looking_flag2
                                    inspection = 1
                                if inspection == 0:
                                    ethogram_vector[init_trial + i] = 2
                            if speed[i] <= SPEED_LIM and closeness == 0 and inspection == 0:
                                ethogram_vector[init_trial + i] = 1

        output_tracking_file = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_day_' + \
                                f'{day + 1}' + '_likelihood_0.75_ethogram.npy'
        output_tracking_path = category_path + output_tracking_file
        np.save(output_tracking_path, ethogram_vector)

        output_tracking_file = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_day_' + \
                               f'{day + 1}' + '_likelihood_0.75_object_corners.npy'
        output_tracking_path = category_path + output_tracking_file
        np.save(output_tracking_path, corners_vector)

        output_speed_file = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_day_' + \
                               f'{day + 1}' + '_likelihood_0.75_speed.npy'
        output_speed_path = category_path + output_speed_file
        np.save(output_speed_path, speed_vector)

        day = day + 1
