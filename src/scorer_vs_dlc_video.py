'''

Created on Thrus 01 Oct 2020
Author: Melisa
This script will create a video with the scorer video and the fancy camera video.

Ww will compare field of view of the videos, crop to the same regios of interest,
align them temporaly, and then add information for dlc to the fancy camera video
to compare the scorer exploration information with what we obtein automatically
from the DLC info and creating an area around the object.

Take into account: sampling rate in fancy camera is 20Hz, while in the scorer video is 30Hz

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

## behaviour directory with information from DLC
behaviour_path = os.environ['DATA_DIR_LOCAL'] + 'compiled_positions/'+f'{mouse}'+'/session_'+ f'{session}'+'/'
## input video path to fancy camera video
input_video_path_dlc = os.environ['DATA_DIR_LOCAL'] + 'videos/Trial1_10072017_2017-07-10-132111-0000.avi'
## input video path to fancy camera video
input_video_path_scorer = os.environ['DATA_DIR_LOCAL'] + 'videos_scorer/mouse_training_OS_calcium_1_t0011.avi'
## output directoy
output_video_path_dlc = os.environ['DATA_DIR_LOCAL'] + 'compare_videos/Trial1_10072017_2017-07-10-132111-0000_dlc.avi'
output_video_path_scorer = os.environ['DATA_DIR_LOCAL'] + 'compare_videos/Trial1_10072017_2017-07-10-132111-0000_scorer.avi'

output_video_path = os.environ['DATA_DIR_LOCAL'] + 'compare_videos/Trial1_10072017_2017-07-10-132111-0000.avi'


## load behaviour from DLC
beh_file_name = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_trial_' + \
                f'{trial}' + '_likelihood_0.75.npy'
beh_path = behaviour_path + beh_file_name
tracking = np.load(beh_path)

## tracking coordinates
x_positions = np.mean(tracking[:, [0, 2, 4, 6, 8]], axis=1).T
y_positions = np.mean(tracking[:, [1, 3, 5, 7, 9]], axis=1).T

#x_position_reshape = np.reshape(x_positions[:int(x_positions.shape[0]/2)*2],(int(x_positions.shape[0]/10),10))
#x_position_resample = np.mean(x_position_reshape,axis =1)

#y_position_reshape = np.reshape(y_positions[:int(y_positions.shape[0]/2)*10],(int(y_positions.shape[0]/10),10))
#y_position_resample = np.mean(y_position_reshape,axis = 1)


## get head direction coordinates
#x_difference = (tracking[:,0] - tracking[:,6]).T
#y_difference = (tracking[:,1] - tracking[:,7]).T
#head_direction = np.array([x_difference , y_difference])
#head_direction = head_direction *10 / npalg.norm(head_direction)

## load input video DLC
if not os.path.isfile(input_video_path_dlc):
    print('ERROR: File not found')
cap_dlc = cv2.VideoCapture(input_video_path_dlc)
try:
    length = int(cap_dlc.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap_dlc.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_dlc.get(cv2.CAP_PROP_FRAME_HEIGHT))
except:
    logging.info('Roll back to opencv 2')
    length = int(cap_dlc.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    width = int(cap_dlc.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(cap_dlc.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
if length == 0 or width == 0 or height == 0:  # CV failed to load
    cv_failed = True
dims_dlc = [length, height, width]
limits = False
ret, frame = cap_dlc.read()
plt.imshow(frame)

### create a new video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#fourcc = cv2.VideoWriter_fourcc(*'DIVX')
output_video_dlc = cv2.VideoWriter(output_video_path_dlc, fourcc, 1, (width ,height))
# Center coordinates
center_coordinates1 = (650,600)
center_coordinates2 = (225,200)
# Radius of circle
radius = 100
# Blue color in BGR
color1 = (255, 0, 0)
color2 = (0, 0, 255)
# Line thickness of 2 px
thickness = 5

time = 0
while True:
    ret, frame = cap_dlc.read()
    if not ret:
        break
    if time % 2 == 0:
        position_vector = np.array([x_positions[int(time/2)],y_positions[int(time/2)]])
        distance1 = npalg.norm(position_vector - center_coordinates1)
        distance2 = npalg.norm(position_vector - center_coordinates2)
        if distance1 < 100:
            cv2.circle(frame,center_coordinates1,radius,color1,thickness)
        if distance2 < 100:
            cv2.circle(frame,center_coordinates2,radius,color2,thickness)
    cv2.waitKey(0)
    if time > 2*20 and time % 20 == 0:
        output_video_dlc.write(frame)
    #print(time)
    time = time + 1


## load input video scorer
if not os.path.isfile(input_video_path_dlc):
    print('ERROR: File not found')
cap_scorer = cv2.VideoCapture(input_video_path_scorer)
try:
    length = int(cap_scorer.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap_scorer.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_scorer.get(cv2.CAP_PROP_FRAME_HEIGHT))
except:
    logging.info('Roll back to opencv 2')
    length = int(cap_scorer.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    width = int(cap_scorer.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(cap_scorer.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
if length == 0 or width == 0 or height == 0:  # CV failed to load
    cv_failed = True
dims_scorer = [length, height, width]
limits = False
ret, frame = cap_scorer.read()
plt.imshow(frame)
plt.show()

### create a new video with proper sampling rate
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#fourcc = cv2.VideoWriter_fourcc(*'DIVX')
output_video_scorer = cv2.VideoWriter(output_video_path_scorer, fourcc, 1, (width ,height))
time = 0
while True:
    ret, frame = cap_scorer.read()
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    if not ret:
        break
    cv2.waitKey(0)
    if time % 17 ==0:
        output_video_scorer.write(frame)
    #print(time)
    time = time + 1

##############################################################################
# Now we use the new created videos with sampling frequency 1fps
##############################################################################

input_video_path_dlc = os.environ['DATA_DIR_LOCAL'] + 'compare_videos/Trial1_10072017_2017-07-10-132111-0000_dlc.avi'
input_video_path_scorer = os.environ['DATA_DIR_LOCAL'] + 'compare_videos/Trial1_10072017_2017-07-10-132111-0000_scorer.avi'


if not os.path.isfile(input_video_path_scorer):
    print('ERROR: File not found')
cap_scorer = cv2.VideoCapture(input_video_path_scorer)
try:
    length = int(cap_scorer.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap_scorer.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_scorer.get(cv2.CAP_PROP_FRAME_HEIGHT))
except:
    logging.info('Roll back to opencv 2')
    length = int(cap_scorer.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    width = int(cap_scorer.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(cap_scorer.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
if length == 0 or width == 0 or height == 0:  # CV failed to load
    cv_failed = True
dims_scorer = [length, height, width]


if not os.path.isfile(input_video_path_dlc):
    print('ERROR: File not found')
cap_dlc = cv2.VideoCapture(input_video_path_dlc)
try:
    length = int(cap_dlc.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap_dlc.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_dlc.get(cv2.CAP_PROP_FRAME_HEIGHT))
except:
    logging.info('Roll back to opencv 2')
    length = int(cap_dlc.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    width = int(cap_dlc.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(cap_dlc.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
if length == 0 or width == 0 or height == 0:  # CV failed to load
    cv_failed = True
dims_dlc = [length, height, width]

### create a new video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#fourcc = cv2.VideoWriter_fourcc(*'DIVX')
h = max(dims_dlc[1], dims_scorer[1])
w = dims_dlc[2] + dims_dlc[2]
output_video = cv2.VideoWriter(output_video_path, fourcc, 1, (w ,h))

time = 0
while time < 300:
    ret1, frame1 = cap_dlc.read()
    ret2, frame2 = cap_scorer.read()
    if not ret1 or not ret2:
        break

    new_frame2 = np.zeros_like(frame1)
    new_frame2[:dims_scorer[1],:dims_scorer[2]] = frame2

    #final_image = np.zeros((h, w, 3))
    #final_image[0:dims_dlc[1],0:dims_dlc[2],:] = frame1
    #final_image[0:dims_scorer[1],dims_dlc[2]:w,:] = frame2

    #final_image = np.concatenate((frame1,new_frame2), axis = 1)
    #cv2.waitKey(0)
    final_image = cv2.hconcat([frame1, new_frame2])
    output_video.write(final_image)
    #print(time)
    time = time + 1

output_video.release()




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

