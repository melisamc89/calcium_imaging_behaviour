'''

Created on Fri 21 Feb 2020
Author: Melisa
'''


import os
import src.configuration
import pandas as pd
import numpy as np
import pickle
import datetime

current_directory = os.environ['PROJECT_DIR'] + 'data/scoring_sheets/'
mice_directory = '56165-56166/'
mouse = 56165
session = 1
min_event_duration = 10

## initial file that conteins mouse, session, trial, resting, and timestramp information. This table conteins all mice info
list_file_name = current_directory + mice_directory+ 'Mouse_c57bl6_calcium_1.xlsx'
file_list_structure = ['condition', 'goal','group','session','drug','subject', 'trial','day', 'loc_1','loc_2']
data = pd.read_excel(list_file_name)
data = pd.DataFrame(data,columns=file_list_structure)
current_data = data.query('subject == ' + f'{mouse}')
current_data = current_data.query('session == ' + f'{session}')
objects = ['LR', 'LL', 'UR', 'UL']
objects_matrix = np.zeros((4,21))

for i in range(len(current_data)):
    if current_data.iloc[i]['loc_1'] == 'LR' or current_data.iloc[i]['loc_2'] == 'LR':
        objects_matrix[0,i] = 1
    if current_data.iloc[i]['loc_1'] == 'LL' or current_data.iloc[i]['loc_2'] == 'LL':
        objects_matrix[1,i] = 1
    if current_data.iloc[i]['loc_1'] == 'UR' or current_data.iloc[i]['loc_2'] == 'UR':
        objects_matrix[2,i] = 1
    if current_data.iloc[i]['loc_1'] == 'UL' or current_data.iloc[i]['loc_2'] == 'UL':
        objects_matrix[3,i] = 1

directory = os.environ['PROJECT_DIR'] + 'data/object_positions/'
file_name = 'mouse_'+f'{mouse}'+'_session_'+f'{session}'+'.npy'
np.save(directory + file_name, objects_matrix)

overlapping_matrix = np.zeros((21,21))

for i in range(objects_matrix.shape[1]):
    for j in range(objects_matrix.shape[1]):
        overlapping_matrix[i,j] = np.dot(objects_matrix[:,i],objects_matrix[:,j])

#figure , axes = plt.subplots(1)
#x = axes.imshow(overlapping_matrix)
#figure.colorbar(x, ax=axes)
#figure.show()

directory = os.environ['PROJECT_DIR'] + 'data/object_positions/'
file_name = 'overlapping_mouse_'+f'{mouse}'+'_session_'+f'{session}'+'.npy'
np.save(directory + file_name, overlapping_matrix)

#%% contions matrix

condition_vector = np.zeros((21,1))
condition_matrix = np.zeros((21,21))
for i in range(objects_matrix.shape[1]):
    if objects_matrix[0,i] == 1:
        if objects_matrix[1,i] == 1:
            condition_vector[i] = 1
        else:
            if objects_matrix[2,i] == 1:
                condition_vector[i] = 2
            else:
                condition_vector[i] = 3
    else:
        if  objects_matrix[1,i] == 1:
            if objects_matrix[2,i] == 1:
                condition_vector[i] = 4
            else:
                condition_vector[i] = 5
        else:
            condition_vector[i] = 6

for i in range(condition_vector.shape[0]):
    for j in range(condition_vector.shape[0]):
            if condition_vector[j] == condition_vector[i]:
                condition_matrix[i,j] = 1


directory = os.environ['PROJECT_DIR'] + 'data/object_positions/'
file_name = 'condition_vector_mouse_'+f'{mouse}'+'_session_'+f'{session}'+'.npy'
np.save(directory + file_name, condition_vector)
file_name = 'condition_matrix_mouse_'+f'{mouse}'+'_session_'+f'{session}'+'.npy'
np.save(directory + file_name, condition_matrix)
