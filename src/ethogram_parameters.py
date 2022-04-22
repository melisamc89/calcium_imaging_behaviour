
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

def objects_position(current_object_data = None):
    ## load objects positions for this trial
    object1 = current_object_data.iloc[session_trial[day][trial]-1]['loc_1']
    object2 = current_object_data.iloc[session_trial[day][trial]-1]['loc_2']
    ## define coordinates of objects in pixels acording to the frame size

    coordinates = [np.array([550, 100]), np.array([150, 100]), np.array([150, 500]), np.array([550, 500])]

    for i in range(4):
        for j in range(4):
            if object1 == objects[i]:
                coordinates1 = coordinates[i]
            if object2 == objects[j]:
                coordinates2 = coordinates[j]

    return [coordinates1, coordinates2]

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

def get_parameters(tracking, coordinates1, coordinates2):

    positions_list = [beh_func.interpolate_positions(signal1=tracking[:, i * 2].T, signal2=tracking[:, i * 2 + 1].T) for
                      i in range(int(tracking.shape[1] / 2))]
    positions = np.array(positions_list)

    cm = np.mean(positions, axis=0)
    d1 = [distance(cm[:, i], coordinates1[i,:]) for i in range(cm.shape[1])]
    d2 = [distance(cm[:, i], coordinates2[i,:]) for i in range(cm.shape[1])]
    d1 = np.array(d1)
    d2 = np.array(d2)

    vx = np.zeros((cm.shape[1], 1))
    vy = np.zeros((cm.shape[1], 1))
    vx[1:, 0] = np.diff(cm[0, :])
    vy[1:, 0] = np.diff(cm[1, :])
    speed = np.squeeze(np.sqrt(vx * vx + vy * vy))

    hd_list = [head_direction(positions[:, :, i]) for i in range(positions.shape[2])]
    hd = np.array(hd_list)

    ## get object one direction
    x_difference = [coordinates1[i, 0] - tracking[i,0] for i in range(tracking.shape[0])]
    y_difference = [coordinates1[i, 1] - tracking[i,1] for i in range(tracking.shape[0])]
    direction = np.array([np.array(x_difference), np.array(y_difference)]).T
    direction = direction / npalg.norm(direction)
    angle1_list = [angle_between(hd[i], direction[i,:]) for i in range(len(hd))]
    angle1 = np.array(angle1_list)

    x_difference = [coordinates2[i, 0] - tracking[i,0] for i in range(tracking.shape[0])]
    y_difference = [coordinates2[i, 1] - tracking[i,1] for i in range(tracking.shape[0])]
    direction2 = np.array([np.array(x_difference), np.array(y_difference)]).T
    direction2 = direction2 / npalg.norm(direction2)
    angle2_list = [angle_between(hd[i], direction2[i,:]) for i in range(len(hd))]
    angle2 = np.array(angle2_list)

    parameters = {'cm': cm, 'head_direction': hd, 'speed': speed, 'angle1': angle1, 'angle2': angle2, 'dist_obj1': d1, 'dist_obj2': d2}

    return parameters


## select mouse and session to analyze
mouse = 32365

## object positions directory
current_directory = os.environ['DATA_DIR_LOCAL'] + 'calcium_imaging_behaviour/data/scoring_sheets/'
if mouse == 32363 or mouse == 32364 or mouse == 32365 or mouse == 32366:
    mice_directory = '32363-32366/'
else:
    mice_directory = '56165-56166/'

for session in [2,3]:

    ## source extracted calcium traces directory
    calcium_directory = os.environ['DATA_DIR_LOCAL'] + 'calcium_imaging_behaviour/data/calcium_activity_day_wise/'
    ## timeline directory
    timeline_file_dir = os.environ['DATA_DIR_LOCAL'] + 'calcium_imaging_behaviour/data/timeline/'
    ## behaviour directory
    behaviour_path = os.environ[
                         'DATA_DIR_LOCAL'] + 'calcium_imaging_behaviour/data/compiled_positions/' + f'{mouse}' + '/session_' + f'{session}' + '/'
    ## output directoy
    category_path = os.environ['DATA_DIR_LOCAL'] + 'calcium_imaging_behaviour/data/ethogram/' + f'{mouse}' + '/session_' + f'{session}' + '/'
    ## initial file that conteins mouse, session, trial, resting, and timestramp information.
    # This table conteins all mice info
    objects_file_name = current_directory + mice_directory + 'mouse_training_OS_calcium_1.xlsx'
    objects_list_structure = ['condition', 'goal', 'group', 'session', 'drug', 'subject', 'trial', 'day', 'loc_1',
                              'loc_2']
    object_list = pd.read_excel(objects_file_name)
    object_list = pd.DataFrame(object_list, columns=objects_list_structure)
    current_object_data = object_list[object_list.subject == mouse]
    current_object_data = current_object_data[current_object_data.session == session]
    objects = ['LR', 'LL', 'UR', 'UL']

    ## behaviour directory
    behaviour_path = os.environ['PROJECT_DIR_LOCAL'] + 'data/compiled_positions/'+f'{mouse}'+'/session_'+ f'{session}'+'/'
    ## output directoy
    category_path = os.environ['PROJECT_DIR_LOCAL']  + 'data/ethogram_parameters/'+f'{mouse}'+'/session_'+ f'{session}'+'/'


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
        parameters_matrix = np.zeros((11,activity.shape[1]))
        parameters_matrix[0,:] = np.arange(0,activity.shape[1])

        ## load tracking of behaviour
        for trial in range(len(session_trial[day])):

            # load objects positions for this trial
            [coor1,coor2] = objects_position(current_object_data)
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
                trial_vector = trial * np.ones_like(tracking[:,0])

                coordinates1 = coor1 * np.ones_like(tracking[:,0:1])
                coordinates2 = coor2 * np.ones_like(tracking[:,0:1])
                parameters = get_parameters(tracking, coordinates1, coordinates2)
                parameters_matrix_trial = np.zeros((10, parameters['cm'].shape[1]))
                parameters_matrix_trial [0, :] = trial_vector
                parameters_matrix_trial [1, :] = parameters['cm'][0, :]
                parameters_matrix_trial [2, :] = parameters['cm'][1, :]
                parameters_matrix_trial [3, :] = parameters['speed']
                parameters_matrix_trial [4, :] = parameters['head_direction'][:, 0]
                parameters_matrix_trial [5, :] = parameters['head_direction'][:, 1]
                parameters_matrix_trial [6, :] = parameters['dist_obj1']
                parameters_matrix_trial [7, :] = parameters['dist_obj2']
                parameters_matrix_trial [8, :] = parameters['angle1']
                parameters_matrix_trial [9, :] = parameters['angle2']

                parameters_matrix[1:11,init_trial : init_trial + parameters_matrix_trial.shape[1]] = parameters_matrix_trial

        output_tracking_file = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_trial_' + \
                            f'{day+1}' + '_likelihood_0.75_ethogram_parameters.npy'
        output_tracking_path = category_path + output_tracking_file
        sciio.savemat(category_path + 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_trial_' + \
                      f'{day + 1}' + '_likelihood_0.75_ethogram_parameters.mat', {'ethogram': parameters_matrix})
        np.save(output_tracking_path,parameters_matrix)
        day = day+1