'''

Created on Fri 07 Feb 2020

Author: Melisa

Lets try to create a behavioural timeline (with the scoring) syncronized with the calcium videos.

'''


import os
import src.configuration
import pandas as pd
import numpy as np
import pickle
import datetime

current_directory = os.environ['PROJECT_DIR'] + 'data/scoring_sheets/'
mice_directory = '32363-32366/'
mouse = 32363
session = 2
min_event_duration = 20

## initial file that conteins mouse, session, trial, resting, and timestramp information. This table conteins all mice info
list_file_name = current_directory + 'mouse_filelist.xlsx'
file_list_structure = ['condition', 'session' , 'mouse', 'date' , 'timestamp','is_rest','trial']
data = pd.read_excel(list_file_name)
data = pd.DataFrame(data,columns=file_list_structure)
current_data = data.query('mouse == ' + f'{mouse}')
current_data = current_data.query('session == ' + f'{session}')
initial_time = current_data['timestamp']

### log file that conteins scoring
log_file_name = current_directory + mice_directory + 'mouse_training_OS_calcium_'+f'{session}' +'_log.xlsx'
file_structure = ['wall_time', 'trial_time', 'frame', 'sequence_nr', 'type', 'start_stop']
table_log = pd.read_excel(log_file_name)
table_log = pd.DataFrame(table_log,columns=file_structure)
events_type = ['TR', 'LR', 'LL', 'UR', 'UL']
start_frame = np.zeros((1,42))
type_of_event = []


## results file from where sequence infomation is taken

path_results = current_directory + mice_directory + 'mouse_training_OS_calcium_'+f'{session}' +'_results.xlsx'
results_file = pd.read_excel(path_results)
mouse_sequence = []
for i in range(len(results_file)):
    x= results_file.iloc[i]['subject']
    y = results_file.iloc[i]['session']
    if x == mouse and y == session:
        mouse_sequence.append(results_file.iloc[i]['sequence_nr'])

session_trial = np.arange(1,22)

event = []
boolean = []
frame = []
trial = []
counter = 0
for i in range(len(table_log)):
    if table_log.iloc[i]['sequence_nr'] in mouse_sequence:
        if table_log.iloc[i]['trial_time'] == 0:
            time_val = int(table_log.iloc[i]['wall_time'])
            time_convert = datetime.datetime.fromtimestamp(time_val)
            hour = time_convert.hour
            minutes = time_convert.minute
            seconds = time_convert.second
            total_sec_behaviour = seconds + minutes * 60 + hour * 360

            time_calcium = str(int(initial_time.iloc[counter]))
            total_sec_calcium = int(time_calcium[-2:]) + int(time_calcium[-4:-2]) * 60 + int(time_calcium[:-4]) * 360
            counter += 1
            sync_diff = round((total_sec_calcium - total_sec_behaviour) / 60)
        new_frame = int((table_log.iloc[i]['trial_time']-sync_diff)*10)
        if new_frame > 0:
            frame.append(new_frame)
            index = mouse_sequence.index(table_log.iloc[i]['sequence_nr'])
            trial.append(index + 1)
            event.append(table_log.iloc[i]['type'])
            boolean.append(table_log.iloc[i]['start_stop'])




## load source extracted calcium traces
file_directory = os.environ['PROJECT_DIR'] + 'data/calcium_activity/'
file_name = 'mouse_'+f'{mouse}'+'_session_'+f'{session}'+'_trial_1_v1.4.100.1.0.1.1.1.npy'
timeline_file_dir = os.environ['PROJECT_DIR'] + 'data/timeline/'
timeline_file_path = timeline_file_dir +  'mouse_'+f'{mouse}'+'_session_'+f'{session}'+'_trial_1_v1.1.1.0.pkl'

activity = np.load(file_directory + file_name)
timeline_file= open(timeline_file_path,'rb')
timeline_info = pickle.load(timeline_file)

timeline = np.zeros(42+1)
for i in range(42):
    timeline[i] = timeline_info[i][1]
timeline[42] = activity.shape[1]
trial_duration = np.diff(timeline)


behavioural_vector = []
trial = np.array(trial)
event = np.array(event)
time = np.array(frame)

for i in range(len(session_trial)):
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
file_name = 'mouse_'+f'{mouse}'+'_session_'+f'{session}'+'_event_'+f'{min_event_duration}' +'.npy'
np.save(directory + file_name, behaviour)

