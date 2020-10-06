'''

Created on Mon 05 Oct 2020
Author: Melisa

In this script we will create a video that takes into account where the mouse is looking at.

Will consider as exploration if the mouse is within a 300 pixels distance to the object,
and of it is also looking at it!

whether the mouse is looking at the object is the harder thing to define.

We will define that as if the two segments that refer to the object and head direction
intersects and that intersection is in a radius of 100 pixels around the objects position.

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
import math

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
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

## select mouse and session to analyze
mouse = 32363
session = 1
trial = 4

## behaviour directory with information from DLC
behaviour_path = os.environ['DATA_DIR_LOCAL'] + 'compiled_positions/'+f'{mouse}'+'/session_'+ f'{session}'+'/'
## input video path to fancy camera video
input_video_path_dlc = os.environ['DATA_DIR_LOCAL'] + 'videos/Trial4_10072017_2017-07-10-140428-0000.avi'
## output directoy
output_video_path_dlc = os.environ['DATA_DIR_LOCAL'] + 'compare_videos/Trial4_10072017_2017-07-10-140428-0000_dlc_new.avi'


## load behaviour from DLC
beh_file_name = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_trial_' + \
                f'{trial}' + '_likelihood_0.75.npy'
beh_path = behaviour_path + beh_file_name
tracking = np.load(beh_path)

## tracking coordinates
x_positions = np.mean(tracking[:, [0, 2, 4, 6, 8]], axis=1).T
y_positions = np.mean(tracking[:, [1, 3, 5, 7, 9]], axis=1).T

## objects positions for this particular video
center_coordinates1 = np.array([225,200])
center_coordinates2 = np.array([225,600])

## get points coordinates
p2 = np.array([tracking[:,0],tracking[:,1]]).T
p1 = np.array([tracking[:,6],tracking[:,7]]).T

p3= center_coordinates1*np.ones_like(p1)
p4 = center_coordinates2*np.ones_like(p1)

## get head direction coordinates
x_difference_head = (tracking[:,0] - tracking[:,6]).T
y_difference_head = (tracking[:,1] - tracking[:,7]).T
head_direction = np.array([x_difference_head , y_difference_head]).T
head_direction = head_direction / npalg.norm(head_direction)

## get object one direction
x_difference_1 = (p3[:,0] - tracking[:,6]).T
y_difference_1 = (p3[:,1] - tracking[:,7]).T
direction1 = np.array([x_difference_1 , y_difference_1]).T
direction1 = direction1 / npalg.norm(direction1)

## get object two direction
x_difference_2 = (p4[:,0] - tracking[:,6]).T
y_difference_2 = (p4[:,1] - tracking[:,7]).T
direction2 = np.array([x_difference_2 , y_difference_2]).T
direction2 = direction2 / npalg.norm(direction2)

## get angle bewtween head direction and object positions
## and assign looking_at_vector
looking_vector1 = np.zeros((p1.shape[0],1))
looking_vector2 = np.zeros((p1.shape[0],1))

angle1_vector = np.zeros((p1.shape[0],1))
angle2_vector = np.zeros((p1.shape[0],1))

for i in range(looking_vector1.shape[0]):
    angle1 = angle_between(head_direction[i], direction1[i])
    angle2 = angle_between(head_direction[i], direction2[i])
    angle1_vector[i] = angle1
    angle2_vector[i] = angle2
    if angle1 < math.pi/4:
        looking_vector1[i,0]=1
    else:
        if angle2 < math.pi/4:
            looking_vector2[i,0] = 1


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
output_video_dlc = cv2.VideoWriter(output_video_path_dlc, fourcc, 10, (width ,height))
# Radius of circle
radius = 150
radius2= 75
## objects positions for this particular video
center_coordinates1 = (225,200)
center_coordinates2 = (225,600)
# Blue color in BGR
color1 = (255, 0, 0)
color2 = (0, 0, 255)
color3 = (0, 255, 0)

# Line thickness of 2 px
thickness = 5
pos_cero = np.array([0,0])

time = 0
while True:
    ret, frame = cap_dlc.read()
    if not ret:
        break
    if time % 2 == 0:
        position_vector = np.array([x_positions[int(time/2)],y_positions[int(time/2)]])
        distance1 = npalg.norm(position_vector - center_coordinates1)
        distance2 = npalg.norm(position_vector - center_coordinates2)
        if position_vector[0] != 0 and position_vector[1] != 0:
            if distance1 < radius and not math.isnan(angle1_vector[int(time/2)]):
                cv2.circle(frame,center_coordinates1,radius,color2,thickness)
                if looking_vector1[int(time/2)] or distance1 < radius2:
                    pt1 = (int(x_positions[int(time/2)]), int(y_positions[int(time/2)]))
                    pt2 = (int(x_positions[int(time/2)] + x_difference_head[int(time/2)]),
                            int(y_positions[int(time/2)] + y_difference_head[int(time/2)]))
                    cv2.arrowedLine(frame, pt1, pt2, (0, 255, 0), 10, 8)
                    cv2.circle(frame,center_coordinates1,radius,color3,thickness)
                    if distance1 < radius2:
                        cv2.circle(frame, center_coordinates1, radius2, color3, thickness)
            else:
                if distance2 < radius  and not math.isnan(angle2_vector[int(time/2)]):
                    cv2.circle(frame, center_coordinates2, radius, color2, thickness)
                    if looking_vector2[int(time / 2)] or distance2 < radius2:
                        pt1 = (int(x_positions[int(time / 2)]), int(y_positions[int(time / 2)]))
                        pt2 = (int(x_positions[int(time / 2)] + x_difference_head[int(time / 2)]),
                               int(y_positions[int(time / 2)] + y_difference_head[int(time / 2)]))
                        cv2.arrowedLine(frame, pt1, pt2, (0, 255, 0), 10, 8)
                        cv2.circle(frame, center_coordinates2, radius, color3, thickness)
                        if distance2 < radius2:
                            cv2.circle(frame, center_coordinates2, radius2, color3, thickness)
                else:
                    if looking_vector1[int(time/2)] and not looking_vector2[int(time/2)]:
                        pt1 = (int(x_positions[int(time/2)]), int(y_positions[int(time/2)]))
                        pt2 = (int(x_positions[int(time/2)] + x_difference_head[int(time/2)]),
                                int(y_positions[int(time/2)] + y_difference_head[int(time/2)]))
                        cv2.arrowedLine(frame, pt1, pt2, (255, 0, 0), 10, 8)
                        cv2.circle(frame,center_coordinates1,radius,color1,thickness)
                    else:
                        if looking_vector2[int(time/2)] and not looking_vector1[int(time/2)]:
                            pt1 = (int(x_positions[int(time/2)]), int(y_positions[int(time/2)]))
                            pt2 = (int(x_positions[int(time/2)] + x_difference_head[int(time/2)]),
                                   int(y_positions[int(time/2)] + y_difference_head[int(time/2)]))
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

