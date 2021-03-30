'''
Created on Monday 15 March 2021 16.42
Author: Melisa
This script is the first attempt to use hmm to analyze states of behaviour in object space task
'''

import numpy as np
import os
import autograd.numpy as np
import cv2
import logging
import autograd.numpy.random as npr
import src.configuration
npr.seed(0)

import matplotlib
import matplotlib.pyplot as plt

import ssm
from ssm.util import find_permutation


# Make an HMM with the true parameters
#true_hmm = ssm.HMM(K, D, observations="diagonal_gaussian")
#z, y = true_hmm.sample(T)
#z_test, y_test = true_hmm.sample(T)
#true_ll = true_hmm.log_probability(y)

# Fit models
N_sgd_iters = 1000
N_em_iters = 100

# A bunch of observation models that all include the
# diagonal Gaussian as a special case.
observations = [
    "diagonal_gaussian",
    "gaussian"
]

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
x = np.arange(len(x_positions))
y = np.arange(len(y_positions))
for i in range(x_positions.shape[0]):
    x[i] = int(x_positions[i])
    y[i] = int(y_positions[i])

diffx = tracking[:,7] - tracking[:,0]
diffy = tracking[:,8] - tracking[:,1]

velx = np.diff(x)
vely = np.diff(y)

velx

#X = tracking[:, [0, 1, 6, 7, 8, 9]]
#X =np.array([x,y,diffx,diffy]).T
X =np.array([x,y,diffx,diffy, velx, vely]).T


# Set the parameters of the HMM
T = X.shape[0]      # number of time bins
K = 5       # number of discrete states
D = X.shape[1]       # number of observed dimensions


# Fit with both SGD and EM
#methods = ["sgd", "em"]
methods = ["em"]

results = {}
for obs in observations:
    for method in methods:
        print("Fitting {} HMM with {}".format(obs, method))
        model = ssm.HMM(K, D, observations=obs)
        train_lls = model.fit(X, method=method)
        #test_ll = model.log_likelihood(y_test)
        smoothed_X = model.smooth(X)

        # Permute to match the true states
        #model.permute(find_permutation(z, model.most_likely_states(y)))
        smoothed_z = model.most_likely_states(X)
        results[(obs, method)] = (model, train_lls, smoothed_z, smoothed_X)


# Plot the inferred states
fig, axs = plt.subplots(len(observations), 1, figsize=(12, 8))

# Plot the inferred states
for i, obs in enumerate(observations):
    zs = []
    _, _, smoothed_z, _ = results[(obs, method)]
    plt.plot(smoothed_z)
    if i != len(observations) - 1:
        plt.xticks()
    else:
        plt.xlabel("time")
    plt.title(obs)

plt.tight_layout()
plt.show()

## input video path
input_video_path = os.environ['DATA_DIR_LOCAL'] + 'videos/'+ 'Trial1_10072017_2017-07-10-132111-0000.avi'
## output directoy
output_video_path = os.environ['DATA_DIR_LOCAL'] + 'hmm_videos/Trial1_10072017_2017-07-10-132111-0000_3.avi'


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

# Blue color in BGR
color0 = (255, 255, 0)
color1 = (255, 0, 0)
color2 = (0, 0, 255)
color3 = (0, 255, 0)
color4 = (255,255,255)
color_vec = [color0, color1, color2, color3, color4]

radius = 10
time = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if time % 2 == 0:
        time1 = int(time/2)
        cv2.circle(frame,(x[time1],y[time1]), radius, color_vec[int(smoothed_z[time1])], 5)

        plt.imshow(frame)
        #cv2.waitKey(0)
        output_video.write(frame)
        #print(time)
    time = time + 1

output_video.release()
cap.release()
