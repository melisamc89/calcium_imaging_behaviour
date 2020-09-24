'''
Created on Wed 26 Aug  2020
Author: Melisa
Lets try to create a behavioural timeline (with the scoring) syncronized with the calcium videos, for day wise
analysis
'''

import os
import src.configuration
import pandas as pd
import numpy as np
import pickle
import datetime

mouse = 56166
session = 2
min_event_duration = 10

current_directory = os.environ['PROJECT_DIR'] + 'data/scoring_sheets/'
mice_directory = '56165-56166/'
file_name = current_directory + mice_directory + 'Mouse_c57bl6_session_'+f'{session}' +'_log.xlsx'

file_structure = ['wall_time', 'trial_time', 'frame', 'sequence_nr', 'type', 'start_stop']
table = pd.read_excel(file_name)
table = pd.DataFrame(table,columns=file_structure)
events_type = ['TR', 'LR', 'LL', 'UR', 'UL']
start_frame = np.zeros((1,42))
type_of_event = []


file_list = current_directory + mice_directory + 'Mouse_c57bl6_filelist.xlsx'
file_list_structure = ['condition', 'session' , 'mouse', 'date' , 'timestamp','is_rest','trial']
data = pd.read_excel(file_list)
data = pd.DataFrame(data,columns=file_list_structure)
current_data = data.query('mouse == ' + f'{mouse}')
current_data = current_data.query('session == ' + f'{session}')
initial_time = current_data['timestamp']

##for 56165
mouse_sequence = []
if mouse == 56165:
    mouse_sequence.append([1,2,3,4,5])
    mouse_sequence.append([11,12,13,14,15])
    mouse_sequence.append([21,22,23,24,25])
    mouse_sequence.append([31,32,33,34,35])
    mouse_sequence.append([41])

##for 56166
if mouse == 56166:
    mouse_sequence.append([6,7,8,9,10])
    mouse_sequence.append([6,17,18,19,20])
    mouse_sequence.append([26,27,28,29,30])
    mouse_sequence.append([36,37,38,39,40])
    mouse_sequence.append([42])

event_list = []
boolean_list = []
frame_list = []
trial_list = []
for j in range(len(mouse_sequence)):
    event = []
    boolean = []
    frame = []
    trial = []
    counter = 0
    for i in range(len(table)):
        if table.iloc[i]['sequence_nr'] in mouse_sequence[j]:
            if table.iloc[i]['trial_time'] == 0:
                time_val = int(table.iloc[i]['wall_time'])
                time_convert = datetime.datetime.fromtimestamp(time_val)
                hour = time_convert.hour
                minutes = time_convert.minute
                seconds = time_convert.second
                total_sec_behaviour = seconds + minutes * 60 + hour * 360

                time_calcium = str(int(initial_time.iloc[counter]))
                total_sec_calcium = int(time_calcium[-2:]) + int(time_calcium[-4:-2]) * 60 + int(time_calcium[:-4]) * 360
                counter += 1
                sync_diff = round((total_sec_calcium - total_sec_behaviour) / 60)
            new_frame = int((table.iloc[i]['trial_time']-sync_diff)*10)
            if new_frame > 0:
                frame.append(new_frame)
                index = mouse_sequence[j].index(table.iloc[i]['sequence_nr'])
                trial.append(index + 1)
                event.append(table.iloc[i]['type'])
                boolean.append(table.iloc[i]['start_stop'])
    event_list.append(event)
    boolean_list.append(boolean)
    frame_list.append(frame)
    trial_list.append(trial)


## load source extracted calcium traces
file_directory = os.environ['PROJECT_DIR'] + 'data/calcium_activity_day_wise/'
timeline_file_dir = os.environ['PROJECT_DIR'] + 'data/timeline/'

day = 0
timeline_length=[10,10,10,10,2]
session_trial = []
session_trial.append(np.arange(1,6))
session_trial.append(np.arange(1,6))
session_trial.append(np.arange(1,6))
session_trial.append(np.arange(1,6))
session_trial.append(np.arange(1,2))
session = 3

for trial_day in [1,6,11,16,21]:

    file_name = 'mouse_'+f'{mouse}'+'_session_'+f'{session}'+'_trial_'+f'{trial_day}' +'_v1.4.20.3.0.1.1.0.npy'
    timeline_file_path = timeline_file_dir + 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_trial_1_v' + f'{1}' + '.4.' + f'{1}' + \
                          '.' + f'{0}' + '_10.pkl'
    activity = np.load(file_directory + file_name)
    timeline_file= open(timeline_file_path,'rb')
    timeline_info = pickle.load(timeline_file)

    timeline = np.zeros(timeline_length[day]+1)
    for i in range(timeline_length[day]):
        timeline[i] = timeline_info[i][1]
    timeline[len(timeline)-1] = activity.shape[1]
    trial_duration = np.diff(timeline)

    behavioural_vector = []
    trial = np.array(trial_list[day])
    event = np.array(event_list[day])
    time = np.array(frame_list[day])

    for i in range(len(session_trial[day])):
        behaviour_trial = np.ones(int(trial_duration[i*2]))
        event_trial = event[np.where(trial == i+1)[0]]
        time_trial = time[np.where(trial == i+1)[0]]
        event_duration = np.diff(time_trial)
        for j in range(1,len(event_duration),2):
            if event_duration[j] > min_event_duration :
                if event_trial[j] == 'LL':
                    behaviour_trial[time_trial[j] : time_trial[j] + event_duration[j]] = 2
                else:
                    if event_trial[j] == 'LR':
                        behaviour_trial[time_trial[j]: time_trial[j] + event_duration[j]] = 3
                    else:
                        if event_trial[j] == 'UR':
                            behaviour_trial[time_trial[j]: time_trial[j] + event_duration[j]] = 4
                        else:
                            if event_trial[j] == 'UL':
                                behaviour_trial[time_trial[j]: time_trial[j] + event_duration[j]] = 5
        behavioural_vector.append(behaviour_trial)

    behaviour = np.zeros(activity.shape[1])

    for i in range(len(behavioural_vector)):
        behaviour[int(timeline[2*i]):int(timeline[2*i+1])] = behavioural_vector[i]

    directory = os.environ['PROJECT_DIR'] + '/data/scoring_time_vector/'
    file_name = 'mouse_'+f'{mouse}'+'_session_'+f'{session}'+'_day_'+f'{day+1}'+'_event_'+f'{min_event_duration}' +'.npy'
    np.save(directory + file_name, behaviour)
    day = day + 1

