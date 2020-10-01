'''

Created on Mon 28 Sep 2020
Author: Melisa
This script will create a video with the mouse moving, the mean positions and head direction of the animal.
'''


import os
import src.configuration
import pandas as pd
import cv2
import numpy as np
import pickle
import numpy.linalg as npalg
import matplotlib.pyplot as plt
import datetime
import logging

## select mouse and session to analyze
mouse = 32363
session = 1
trial = 1

## behaviour directory
behaviour_path = os.environ['DATA_DIR_LOCAL'] + 'compiled_positions/'+f'{mouse}'+'/session_'+ f'{session}'+'/'
## input video path
input_video_path = os.environ['DATA_DIR_LOCAL'] + 'videos/'+ 'Trial1_10072017_2017-07-10-132111-0000.avi'
## output directoy
output_video_path = os.environ['DATA_DIR_LOCAL'] + 'head_direction_videos/Trial1_10072017_2017-07-10-132111-0000.avi'

## load behaviour from DLC
beh_file_name = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_trial_' + \
                f'{trial}' + '_likelihood_0.75.npy'
beh_path = behaviour_path + beh_file_name
tracking = np.load(beh_path)

## tracking coordinates
x_positions = np.mean(tracking[:, [0, 2, 4, 6, 8]], axis=1).T
y_positions = np.mean(tracking[:, [1, 3, 5, 7, 9]], axis=1).T

## get head direction coordinates
x_difference = (tracking[:,0] - tracking[:,6]).T
y_difference = (tracking[:,1] - tracking[:,7]).T
head_direction = np.array([x_difference , y_difference])
head_direction = head_direction *10 / npalg.norm(head_direction)

## load input video
if not os.path.isfile(input_video_path):
    print('ERROR: File not found')

cap = cv2.VideoCapture(input_video_path)

try:
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
except:
    logging.info('Roll back to opencv 2')
    length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

if length == 0 or width == 0 or height == 0:  # CV failed to load
    cv_failed = True

dims = [length, height, width]
limits = False
ret, frame = cap.read()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#fourcc = cv2.VideoWriter_fourcc(*'DIVX')
output_video = cv2.VideoWriter(output_video_path, fourcc, 10, (width ,height))

time = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if time % 2 == 0:
        time1 = int(time/2)
        pt1 = (int(x_positions[time1]),int(y_positions[time1]))
        pt2 = (int(x_positions[time1] + x_difference[time1]),int(y_positions[time1]+y_difference[time1]))
        cv2.arrowedLine(frame, pt1, pt2, (255, 0, 0), 10, 8)
        plt.imshow(frame)
        #cv2.waitKey(0)
        output_video.write(frame)
        #print(time)
    time = time + 1

output_video.release()
cap.release()




#figure, axes = plt.subplots(1)

#for i in range(x_positions.shape[0]):
#    axes.arrow(x_positions[i], y_positions[i], x_difference[i], y_difference[i])
#    axes.annotate("->", xy=(x_positions[i], y_positions[i]), color = 'b')

#axes.set_xlabel('X [pixels]')
#axes.set_ylabel('Y [pixels]')

#figure.suptitle('Head Direction')
#figure.show()
#figure_path = '/home/melisa/Documents/calcium_imaging_behaviour/figures/'
#figure_file_name = figure_path + 'mouse_32363_session_1_event_10_likelihood_0.75_head_direction.png'
#figure.savefig(figure_file_name)