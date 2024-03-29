'''

Created on Wed 02 Mar 2022
Author: Melisa
Take the *.csv file from DLC and check the tracking
'''

import os
import src.Alonso.configuration
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
cmap = cm.jet

output_figure_path = os.environ['PROJECT_DIR_LOCAL'] + 'figures/'
input_path = os.environ['DATA_DIR_LOCAL'] + '411857/tracking/week1/'

input_file_path = input_path + '20210817_411857_trial1-08172021080504-0000DLC_resnet50_object_space_2021sep9shuffle1_50000.csv'

tracking_data = pd.read_csv(input_file_path)
body_parts = ['nose', 'ear1', 'ear2' , 'head', 'middle_body','tail_start']
body_part_structure = ['x', 'y', 'likelihood']


tracking_data_array = tracking_data.to_numpy()

LIKELIHOOD = 0.75

x_nose = np.round(tracking_data_array[2:,1].astype(np.float),2)
y_nose = np.round(tracking_data_array[2:,2].astype(np.float),2)
likelihood_nose = np.round(tracking_data_array[2:,3].astype(np.float),2)

x_ear1 = np.round(tracking_data_array[2:,4].astype(np.float),2)
y_ear1 = np.round(tracking_data_array[2:,5].astype(np.float),2)
likelihood_ear1 = np.round(tracking_data_array[2:,6].astype(np.float),2)

x_ear2 = np.round(tracking_data_array[2:,7].astype(np.float),2)
y_ear2 = np.round(tracking_data_array[2:,8].astype(np.float),2)
likelihood_ear2 = np.round(tracking_data_array[2:,9].astype(np.float),2)


x_head = np.round(tracking_data_array[2:,10].astype(np.float),2)
y_head = np.round(tracking_data_array[2:,11].astype(np.float),2)
likelihood_head = np.round(tracking_data_array[2:,12].astype(np.float),2)


selection_nose = np.where(likelihood_nose>LIKELIHOOD)
selection_ear1 = np.where(likelihood_ear1>LIKELIHOOD)
selection_ear2 = np.where(likelihood_ear2>LIKELIHOOD)
selection_head = np.where(likelihood_head>LIKELIHOOD)

intersec1 = np.intersect1d(selection_nose,selection_ear1)
intersec2 = np.intersect1d(selection_ear2,selection_head)
selection= np.intersect1d(intersec1,intersec2)

new_x_nose = x_nose[selection]
new_y_nose = y_nose[selection]

new_x_ear1 = x_ear1[selection]
new_y_ear1 = y_ear1[selection]

new_x_ear2 = x_ear2[selection]
new_y_ear2 = y_ear2[selection]

new_x_head = x_head[selection]
new_y_head = y_head[selection]

figure, axes = plt.subplots(1)
#color = np.linspace(0, 20, new_x.shape[0])
#axes.scatter(new_x,new_y, c = color, cmap = cmap)
axes.plot(new_x_nose,new_y_nose)
axes.plot(new_x_ear1,new_y_ear1)
axes.plot(new_x_ear2,new_y_ear2)
axes.plot(new_x_head,new_y_head)

axes.legend(['Nose', 'Ear1', 'Ear2', 'Head'])
axes.set_xlabel('X [pixels]')
axes.set_ylabel('Y [pixels]')
axes.set_xlim([0,1000])
axes.set_ylim([0,1000])
figure.suptitle('Tracking. Likelihood:' + f'{LIKELIHOOD}')

figure.show()
output_figure_file_path = output_figure_path + '20210817_411857_trial1-08172021080504-0000DLC_resnet50_object_space_2021sep9shuffle1_50000_' + f'{LIKELIHOOD}' + '.png'
figure.savefig(output_figure_file_path)