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


def direction(p3= None, p4= None,p1 = None):
    d = np.cross((p1-p3),(p4-p3))
    return np.sign(d)


# checks if line segment p1p2 and p3p4 intersect
def intersect(p1, p2, p3, p4):
    d1 = direction(p3, p4, p1)
    d2 = direction(p3, p4, p2)
    d3 = direction(p1, p2, p3)
    d4 = direction(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
        ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    else:
        return False

    #elif d1 == 0 and on_segment(p3, p4, p1):
    #    return True
    #elif d2 == 0 and on_segment(p3, p4, p2):
    #    return True
    #elif d3 == 0 and on_segment(p1, p2, p3):
    #    return True
    #elif d4 == 0 and on_segment(p1, p2, p4):
    #    return True
    #else:
    #    return False


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
output_video_path_dlc = os.environ['DATA_DIR_LOCAL'] + 'compare_videos/Trial1_10072017_2017-07-10-132111-0000_dlc_new.avi'


## load behaviour from DLC
beh_file_name = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_trial_' + \
                f'{trial}' + '_likelihood_0.75.npy'
beh_path = behaviour_path + beh_file_name
tracking = np.load(beh_path)

## tracking coordinates
x_positions = np.mean(tracking[:, [0, 2, 4, 6, 8]], axis=1).T
y_positions = np.mean(tracking[:, [1, 3, 5, 7, 9]], axis=1).T

## objects positions for this particular video
center_coordinates1 = np.array([650,600])
center_coordinates2 = np.array([225,200])

## get head direction coordinates
x_difference = (tracking[:,0] - tracking[:,6]).T
y_difference = (tracking[:,1] - tracking[:,7]).T
head_direction = np.array([x_difference , y_difference])
head_direction = head_direction *10 / npalg.norm(head_direction)

## get points coordinates
p2 = 10*np.array([tracking[:,0],tracking[:,1]]).T
p1 = np.array([tracking[:,6],tracking[:,7]]).T

position_cero0 = np.array([0,800])
position_cero1 = np.array([800,800])
p3 = position_cero0*np.ones_like(p1)
p3_1 = position_cero1*np.ones_like(p1)
p4 = center_coordinates1*np.ones_like(p1)*1.5
p5 = center_coordinates2*np.ones_like(p1)*1.5

looking_vector1 = np.zeros((p1.shape[0],1))
looking_vector2 = np.zeros((p1.shape[0],1))
for i in range(looking_vector1.shape[0]):
    if intersect(p1[i], p2[i], p3[i], p4[i]): # and intersect(p1[i], p2[i], p3_1[i], p4[i]):
        looking_vector1[i,0]=1
    if intersect(p1[i], p2[i], p3[i], p5[i]):# and intersect(p1[i], p2[i], p3_1[i], p5[i]):
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
output_video_dlc = cv2.VideoWriter(output_video_path_dlc, fourcc, 20, (width ,height))
# Radius of circle
radius = 200
## objects positions for this particular video
center_coordinates1 = (650,600)
center_coordinates2 = (225,200)
# Blue color in BGR
color1 = (255, 0, 0)
color2 = (0, 0, 255)
color3 = (0, 255, 0)

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
        if distance1 < radius:
            cv2.circle(frame,center_coordinates1,radius,color2,thickness)
            if looking_vector1[int(time/2)]:
                pt1 = (int(x_positions[int(time/2)]), int(y_positions[int(time/2)]))
                pt2 = (int(x_positions[int(time/2)] + x_difference[int(time/2)]),
                        int(y_positions[int(time/2)] + y_difference[int(time/2)]))
                cv2.arrowedLine(frame, pt1, pt2, (0, 255, 0), 10, 8)
                cv2.circle(frame,center_coordinates1,radius,color3,thickness)
        else:
            if distance2 < radius:
                cv2.circle(frame, center_coordinates2, radius, color2, thickness)
                if looking_vector2[int(time / 2)]:
                    pt1 = (int(x_positions[int(time / 2)]), int(y_positions[int(time / 2)]))
                    pt2 = (int(x_positions[int(time / 2)] + x_difference[int(time / 2)]),
                           int(y_positions[int(time / 2)] + y_difference[int(time / 2)]))
                    cv2.arrowedLine(frame, pt1, pt2, (0, 255, 0), 10, 8)
                    cv2.circle(frame, center_coordinates2, radius, color3, thickness)
            else:
                if looking_vector1[int(time/2)] and not looking_vector2[int(time/2)]:
                    pt1 = (int(x_positions[int(time/2)]), int(y_positions[int(time/2)]))
                    pt2 = (int(x_positions[int(time/2)] + x_difference[int(time/2)]),
                            int(y_positions[int(time/2)] + y_difference[int(time/2)]))
                    cv2.arrowedLine(frame, pt2, pt1, (255, 0, 0), 10, 8)
                    cv2.circle(frame,center_coordinates1,radius,color1,thickness)
                else:
                    if looking_vector2[int(time/2)] and not looking_vector1[int(time/2)]:
                        pt1 = (int(x_positions[int(time/2)]), int(y_positions[int(time/2)]))
                        pt2 = (int(x_positions[int(time/2)] + x_difference[int(time/2)]),
                               int(y_positions[int(time/2)] + y_difference[int(time/2)]))
                        cv2.arrowedLine(frame, pt1, pt2, (255, 0, 0), 10, 8)
                        cv2.circle(frame,center_coordinates2,radius,color1,thickness)

    cv2.waitKey(0)
    #if time > 2*20 and time % 20 == 0:
    output_video_dlc.write(frame)
    #print(time)
    time = time + 1


