

import pandas as pd

current_directory = '/home/melisa/Documents/calcium_imaging_behaviour/scoring_sheets/'
mice_directory = '56165-56166/'

file_name = current_directory + mice_directory + 'Mouse_c57bl6_calcium_1_log.xlsx'

file_structure = ['wall_time', 'trial_time', 'frame', 'sequence_nr', 'type', 'start_stop']

table = pd.read_excel(file_name)
table = pd.DataFrame(table,columns=file_structure)

events_type = ['TR', 'LR', 'LL', 'UR', 'UL']
start_frame = np.zeros((1,42))
type_of_event = []
for i in range(len(table)):
    for type in range(len(events)):
        if table.iloc[i].name[4] == events[type]:
            type_of_event.append(0)