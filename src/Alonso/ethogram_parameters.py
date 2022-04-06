
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

    if object1 == 'LL':
        object1_x = 800
        object1_y = 250
    if object1 == 'LR':
        object1_x = 350
        object1_y = 250
    if object1 == 'UR':
        object1_x = 350
        object1_y = 700
    if object1 == 'UL':
        object1_x = 800
        object1_y = 700

    if object2 == 'LL':
        object2_x = 800
        object2_y = 250
    if object2 == 'LR':
        object2_x = 350
        object2_y = 250
    if object2 == 'UR':
        object2_x = 350
        object2_y = 700
    if object2 == 'UL':
        object2_x = 800
        object2_y = 700
    coordinates1 = np.array([object1_x, object1_y])
    coordinates2 = np.array([object2_x, object2_y])

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
mouse = 411857
for session in [1,2,3]:

    ## object positions directory

    current_directory  = os.environ['DATA_DIR_LOCAL'] +f'{mouse}'+'/'

    ## source extracted calcium traces directory
    calcium_directory = os.environ['DATA_DIR_LOCAL'] +f'{mouse}'+'/' + 'data/calcium_activity_day_wise/'
    ## timeline directory
    timeline_file_dir =os.environ['DATA_DIR_LOCAL'] +f'{mouse}'+'/' + 'data/timeline/'

    objects_file_name = current_directory + 'training_sheet_411857.xlsx'
    objects_list_structure = ['condition', 'goal','group','session','drug','subject', 'trial','day', 'loc_1','loc_2']

    object_list = pd.read_excel(objects_file_name)
    object_list = pd.DataFrame(object_list,columns=objects_list_structure)

    #current_object_data = object_list[ object_list.subject == mouse]
    current_object_data =object_list[object_list.session == session]
    objects = ['LR', 'LL', 'UR', 'UL']

    ## behaviour directory
    behaviour_path = os.environ['PROJECT_DIR_LOCAL'] + 'data/compiled_positions/'+f'{mouse}'+'/week'+ f'{session}'+'/'
    ## output directoy
    category_path = os.environ['PROJECT_DIR_LOCAL']  + 'data/ethogram_parameters/'+f'{mouse}'+'/week'+ f'{session}'+'/'


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
        parameters_matrix = np.zeros((9,activity.shape[1]))

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

                coordinates1 = coor1 * np.ones_like(tracking[:,0:1])
                coordinates2 = coor2 * np.ones_like(tracking[:,0:1])
                parameters = get_parameters(tracking, coordinates1, coordinates2)
                parameters_matrix_trial = np.zeros((9, parameters['cm'].shape[1]))
                parameters_matrix_trial [0, :] = parameters['cm'][0, :]
                parameters_matrix_trial [1, :] = parameters['cm'][1, :]
                parameters_matrix_trial [2, :] = parameters['speed']
                parameters_matrix_trial [3, :] = parameters['angle1']
                parameters_matrix_trial [4, :] = parameters['angle2']
                parameters_matrix_trial [5, :] = parameters['head_direction'][:, 0]
                parameters_matrix_trial [6, :] = parameters['head_direction'][:, 1]
                parameters_matrix_trial [7, :] = parameters['dist_obj1']
                parameters_matrix_trial [8, :] = parameters['dist_obj2']

                parameters_matrix[:,init_trial : init_trial + parameters_matrix_trial.shape[1]] = parameters_matrix_trial

        output_tracking_file = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_trial_' + \
                            f'{day+1}' + '_likelihood_0.75_ethogram_parameters.npy'
        output_tracking_path = category_path + output_tracking_file
        sciio.savemat(category_path + 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_trial_' + \
                      f'{day + 1}' + '_likelihood_0.75_ethogram_parameters.mat', {'ethogram': parameters_matrix})
        np.save(output_tracking_path,parameters_matrix)
        day = day+1