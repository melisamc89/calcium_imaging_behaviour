'''

Created on Mon 29 mrt 2021
Author: Melisa
This script will create a video with the mouse moving and ethogram definitions based on ethogram files:

Will generate a discrete vector with information about the moments when the animal is involved in certain ethogram
defined behaviours.

1. Navigation : animal is far away from the object (circle defined as 200px radious) and running at speed > 2cm/s
2. Resting    : animal is far away from the object circle defined as 200px radious and speed < 2cm/s
3. Exploring  : animal is close to object (with a big radius of 200px) and is inspecting it, or it is at a closer distance (radius < 100 px)
4. Inspection : animal is looking at the object (angle between head direction and object direcction < 45')

Exploting and inspection can be divided in the different objects the animal is looking at, given by the positions of the objects.

1. LL : lower left object
2. LR : Lower right object
3. UL : uper left object
4. UR : uper right object

and generates 5 different states :

1. Resting outside arena.
2. Resting inside arena : far away from object and speed < 50px/s
3. Navigation : far way from object and speed > 50px/s and the animal is not looking at the object.
4. Running towards object: speed is >50px/s and animal is looking at object.
5. Exploring object X: animal is either to close to the object (<75 px) or speed is small (<50px/s )
 and distance to object is < 200 px and the animal is looking at the object.
'''


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
import math
import configuration
import behavioural_analysis_functions as beh_func


## select mouse and session to analyze
mouse = 32363
session = 1
trial = 4

## behaviour directory
behaviour_path = os.environ['DATA_DIR_LOCAL'] + 'compiled_positions/'+f'{mouse}'+'/session_'+ f'{session}'+'/'
## input video path to fancy camera video
input_video_path_dlc = os.environ['DATA_DIR_LOCAL'] + 'videos/Trial4_10072017_2017-07-10-140428-0000.avi'
## output directoy
output_video_path_dlc = os.environ['DATA_DIR_LOCAL'] + 'ethogram_videos/Trial4_10072017_2017-07-10-140428-0000.avi'

## load behaviour from DLC (tracking information from csv file)
beh_file_name = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_trial_' + \
                f'{trial}' + '_likelihood_0.75.npy'
beh_path = behaviour_path + beh_file_name
tracking = np.load(beh_path)
## tracking coordinates
## tracking coordinates
x_positions = np.mean(tracking[:, [0, 2, 4, 6, 8]], axis=1).T
y_positions = np.mean(tracking[:, [1, 3, 5, 7, 9]], axis=1).T
position = np.array([x_positions, y_positions]).T
vx = np.zeros((x_positions.shape[0], 1))
vy = np.zeros((y_positions.shape[0], 1))
vx[1:,0]=np.diff(x_positions)
vy[1:,0]=np.diff(y_positions)
position = np.array([x_positions, y_positions]).T
speed = np.sqrt(vx*vx+vy*vy)

## objects positions for this particular video
object1_x = 225
object1_y = 600
object2_x = 225
object2_y = 250
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
proximity_vector1 = beh_func.proximity_vector(position,p3,radius=100)
proximity_vector2 = beh_func.proximity_vector(position,p4,radius=100)

## super proximity vector for mouse position and objects (closer that proximity1)
super_proximity_vector1 = beh_func.proximity_vector(position,p3,radius=50)
super_proximity_vector2 = beh_func.proximity_vector(position,p4,radius=50)

## select events of a certain duration
looking_vector1_last = beh_func.long_duration_events(looking_vector1,5)
looking_vector2_last = beh_func.long_duration_events(looking_vector2,5)

proximity_vector1_last = beh_func.long_duration_events(proximity_vector1,4)
proximity_vector2_last = beh_func.long_duration_events(proximity_vector2,4)

super_proximity_vector1_last = beh_func.long_duration_events(super_proximity_vector1,2)
super_proximity_vector2_last = beh_func.long_duration_events(super_proximity_vector2,2)



####################################################################################################################
# FROM HERE THE SCRIPT IS RELATED TO GENERATE THE VIDEO!
####################################################################################################################
## load input video DLC using cv2
font = cv2.FONT_HERSHEY_SIMPLEX

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
radius = 100
radius2= 50
speed_lim = 5
## objects positions for this particular video
center_coordinates1 = (object1_x,object1_y)
center_coordinates2 = (object2_x,object2_y)
# Blue color in BGR
color1 = (255, 0, 0)
color2 = (0, 0, 255)
color3 = (0, 255, 0)
color4 = (255,255,0)

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
            pt1 = (int(p1[int(time / 2), 0]), int(p1[int(time / 2), 1]))
            pt2 = (int(p2[int(time / 2), 0]), int(p2[int(time / 2), 1]))
            closesness = 0
            inspection = 0
            if proximity_vector1_last[int(time/2)] and not math.isnan(angle1_vector[int(time/2)]):
                cv2.circle(frame,center_coordinates1,radius,color2,thickness)
                closesness = 1
                cv2.arrowedLine(frame, pt1, pt2, color4, 5, 8)
                if looking_vector1_last[int(time/2)] or super_proximity_vector1_last[int(time/2)]:
                    cv2.putText(frame, 'ExploringObj1', (10, 450),font, 3,color4, 2, cv2.LINE_AA)
                    cv2.circle(frame,center_coordinates1,radius,color4,thickness)
                    if super_proximity_vector1_last[int(time/2)]:
                        cv2.circle(frame, center_coordinates1, radius2, color4, thickness)
            if proximity_vector2_last[int(time/2)]  and not math.isnan(angle2_vector[int(time/2)]):
                cv2.circle(frame, center_coordinates2, radius, color4, thickness)
                closesness = 1
                cv2.arrowedLine(frame, pt1, pt2, color4, 5, 8)
                if looking_vector2_last[int(time / 2)] or super_proximity_vector1[int(time/2)]:
                    cv2.circle(frame, center_coordinates2, radius, color4, thickness)
                    cv2.putText(frame, 'ExploringObj2', (10, 450),font, 3,color4, 2, cv2.LINE_AA)
                    if super_proximity_vector2[int(time/2)]:
                        cv2.circle(frame, center_coordinates2, radius2, color4, thickness)
            if speed[int(time/2)] > speed_lim and closesness == 0:
                if looking_vector1_last[int(time/2)] and not looking_vector2_last[int(time/2)]:
                    inspection = 1
                    cv2.arrowedLine(frame, pt1, pt2, color2, 5, 8)
                    cv2.putText(frame, 'RunningTo1',(10, 450),font, 3,color2, 2, cv2.LINE_AA)
                    cv2.circle(frame, (int(position[int(time/2),0]),int(position[int(time/2),1])), radius,
                               color2, thickness)
                if looking_vector2_last[int(time/2)] and not looking_vector1_last[int(time/2)]:
                    inspection = 1
                    cv2.arrowedLine(frame, pt1, pt2, color2, 5, 8)
                    cv2.putText(frame, 'RunningTo2', (10, 450),font, 3,color2, 2, cv2.LINE_AA)
                    cv2.circle(frame, (int(position[int(time / 2), 0]), int(position[int(time / 2), 1])),
                               radius, color2, thickness)
                if inspection == 0:
                    cv2.arrowedLine(frame, pt1, pt2, color3, 5, 8)
                    cv2.putText(frame, 'Running', (10, 450),font, 3,color3, 3, cv2.LINE_AA)
                    cv2.circle(frame, (int(position[int(time / 2), 0]), int(position[int(time / 2), 1])),
                               radius, color3, thickness)
            if speed[int(time/2)] < speed_lim and closesness == 0 and inspection == 0:
                    cv2.arrowedLine(frame, pt1, pt2, color1, 5, 8)
                    cv2.putText(frame, 'Resting',(10, 450),font, 3, color1, 2, cv2.LINE_AA)
                    cv2.circle(frame, (int(position[int(time / 2), 0]), int(position[int(time / 2), 1])), radius,
                                       color1, thickness)

            cv2.waitKey(0)
    #if time > 2*20 and time % 20 == 0:
            output_video_dlc.write(frame)
    #print(time)
    time = time + 1
