'''

Created on Mon 05 Oct 2020
Author: Melisa

In this script we will create a video that takes into account where the mouse is looking at.

Will consider as exploration if the mouse is within a 150/75 pixels distance to the object,
and of it is also looking at it!
Whether the mouse is looking at the object is the harder thing to define. We use the direction between
the head and the objects positions to define it. If the angle between those is smoller that
a certain threshold then the animal is looking at the object. Uses behavioural_analysis_functions

'''

import os
import src.configuration
import pandas as pd
import cv2
import numpy as np
import pickle
import numpy.linalg as npalg
import matplotlib.pyplot as plt
import logging
import math
import src.behavioural_analysis_functions as beh_func

## select mouse and session to analyze
mouse = 32363
session = 1
trial = 4

### define relevant paths
## behaviour directory with information from DLC
behaviour_path = os.environ['DATA_DIR_LOCAL'] + 'compiled_positions/'+f'{mouse}'+'/session_'+ f'{session}'+'/'
## input video path to fancy camera video
input_video_path_dlc = os.environ['DATA_DIR_LOCAL'] + 'videos/Trial4_10072017_2017-07-10-140428-0000.avi'
## output directoy
output_video_path_dlc = os.environ['DATA_DIR_LOCAL'] + 'compare_videos/Trial4_10072017_2017-07-10-140428-0000_dlc_new.avi'

## load behaviour from DLC (tracking information from csv file)
beh_file_name = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_trial_' + \
                f'{trial}' + '_likelihood_0.75.npy'
beh_path = behaviour_path + beh_file_name
tracking = np.load(beh_path)

## tracking coordinates
x_positions = np.mean(tracking[:, [0, 2, 4, 6, 8]], axis=1).T
y_positions = np.mean(tracking[:, [1, 3, 5, 7, 9]], axis=1).T
position = np.array([x_positions, y_positions]).T

## objects positions for this particular video
object1_x = 225
object1_y = 200
object2_x = 225
object2_y = 600
center_coordinates1 = np.array([object1_x,object1_y])
center_coordinates2 = np.array([object2_x,object2_y])

## get points coordinates for head direction and objects location
p2 = np.array([tracking[:,0],tracking[:,1]]).T # nose position
p1 = np.array([tracking[:,6],tracking[:,7]]).T # head position
p3= center_coordinates1*np.ones_like(p1)
p4 = center_coordinates2*np.ones_like(p1)

## binary looking at object vectors
looking_vector1, angle1_vector = beh_func.looking_at_vector(p2,p1,p3)
looking_vector2, angle2_vector = beh_func.looking_at_vector(p2,p1,p4)

## proximity vector between mouse position and objects
proximity_vector1 = beh_func.proximity_vector(position,p3,radius=150)
proximity_vector2 = beh_func.proximity_vector(position,p4,radius=150)

## super proximity vector for mouse position and objects (closer that proximity1)
super_proximity_vector1 = beh_func.proximity_vector(position,p3,radius=75)
super_proximity_vector2 = beh_func.proximity_vector(position,p4,radius=75)

## select events of a certain duration
looking_vector1_last = beh_func.long_duration_events(looking_vector1,10)
looking_vector2_last = beh_func.long_duration_events(looking_vector2,10)

proximity_vector1_last = beh_func.long_duration_events(proximity_vector1,10)
proximity_vector2_last = beh_func.long_duration_events(proximity_vector2,10)

super_proximity_vector1_last = beh_func.long_duration_events(super_proximity_vector1,5)
super_proximity_vector2_last = beh_func.long_duration_events(super_proximity_vector2,5)


####################################################################################################################
# FROM HERE THE SCRIPT IS RELATED TO GENERATE THE VIDEO!
####################################################################################################################
## load input video DLC using cv2

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
output_video_dlc = cv2.VideoWriter(output_video_path_dlc, fourcc, 10, (width ,height))
# Radius of circle
radius = 150
radius2= 75
## objects positions for this particular video
center_coordinates1 = (object1_x,object1_y)
center_coordinates2 = (object2_x,object2_y)
# Blue color in BGR
color1 = (255, 0, 0)
color2 = (0, 0, 255)
color3 = (0, 255, 0)

# Line thickness of 5 px
thickness = 5
pos_cero = np.array([0,0])
time = 0
while True:
    ret, frame = cap_dlc.read()
    if not ret:
        break
    if time % 2 == 0:
        if position[int(time/2),0] != 0 and position[int(time/2),1] != 0:
            if proximity_vector1_last[int(time/2)] and not math.isnan(angle1_vector[int(time/2)]):
                cv2.circle(frame,center_coordinates1,radius,color2,thickness)
                if looking_vector1_last[int(time/2)] or super_proximity_vector1_last[int(time/2)]:
                    pt1 = (int(p1[int(time/2),0]), int(p1[int(time/2),1]))
                    pt2 = (int(p2[int(time/2),0]),int(p2[int(time/2),1]))
                    cv2.arrowedLine(frame, pt1, pt2, (0, 255, 0), 10, 8)
                    cv2.circle(frame,center_coordinates1,radius,color3,thickness)
                    if  super_proximity_vector1_last[int(time/2)]:
                        cv2.circle(frame, center_coordinates1, radius2, color3, thickness)
            else:
                if proximity_vector2_last[int(time/2)]  and not math.isnan(angle2_vector[int(time/2)]):
                    cv2.circle(frame, center_coordinates2, radius, color2, thickness)
                    if looking_vector2[int(time / 2)] or super_proximity_vector1[int(time/2)]:
                        pt1 = (int(p1[int(time / 2), 0]), int(p1[int(time / 2), 1]))
                        pt2 = (int(p2[int(time / 2), 0]), int(p2[int(time / 2), 1]))
                        cv2.arrowedLine(frame, pt1, pt2, (0, 255, 0), 10, 8)
                        cv2.circle(frame, center_coordinates2, radius, color3, thickness)
                        if super_proximity_vector2[int(time/2)]:
                            cv2.circle(frame, center_coordinates2, radius2, color3, thickness)
                else:
                    if looking_vector1_last[int(time/2)] and not looking_vector2_last[int(time/2)]:
                        pt1 = (int(p1[int(time / 2), 0]), int(p1[int(time / 2), 1]))
                        pt2 = (int(p2[int(time / 2), 0]), int(p2[int(time / 2), 1]))
                        cv2.arrowedLine(frame, pt1, pt2, (255, 0, 0), 10, 8)
                        cv2.circle(frame,center_coordinates1,radius,color1,thickness)
                    else:
                        if looking_vector2_last[int(time/2)] and not looking_vector1_last[int(time/2)]:
                            pt1 = (int(p1[int(time / 2), 0]), int(p1[int(time / 2), 1]))
                            pt2 = (int(p2[int(time / 2), 0]), int(p2[int(time / 2), 1]))
                            cv2.arrowedLine(frame, pt1, pt2, (255, 0, 0), 10, 8)
                            cv2.circle(frame,center_coordinates2,radius,color1,thickness)

            cv2.waitKey(0)
    #if time > 2*20 and time % 20 == 0:
            output_video_dlc.write(frame)
    #print(time)
    time = time + 1


####################################################################
### Now compare the two videos
######################################################################
input_video_path_dlc = os.environ['DATA_DIR_LOCAL'] + 'compare_videos/Trial1_10072017_2017-07-10-132111-0000_dlc_new.avi'
input_video_path_scorer = os.environ['DATA_DIR_LOCAL'] + 'compare_videos/Trial1_10072017_2017-07-10-132111-0000_scorer.avi'

output_video_path = os.environ['DATA_DIR_LOCAL'] + 'compare_videos/Trial1_10072017_2017-07-10-132111-0000_comparison_head_direction.avi'

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

