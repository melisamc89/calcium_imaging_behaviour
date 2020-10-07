'''

Created on Wed 07 OCt 2020
Author: Melisa

Functions that will be required to process different types of behaviours in the tracking
'''

import os
import numpy as np
import numpy.linalg as npalg
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

def looking_at_vector(nose_position = None, head_position = None, entity_coordinates = None):

    '''
    Creates a binary vector that states whether the animal is looking towards something.
    The direction of looking is computed with the head direction vector
    , and the other's entity position is computed with the direction of the head and the object
    coordinates position
    :param  noise_postion: 2d X N array containing 2D position of nose
            head_position: 2d X N array containing 2D head position
            entity_coordinates: 2d X N array containing 2D position of entity
    :return: looking_vector: binary vector with information about the fact that the mouse is looking at an object
            angle_vector: angle between head direction and entity's coordinates
    '''

    ## get head direction coordinates
    x_difference_head = (nose_position[:,0] - head_position[:,0]).T
    y_difference_head = (nose_position[:,1] - head_position[:,1]).T
    head_direction = np.array([x_difference_head, y_difference_head]).T
    head_direction = head_direction / npalg.norm(head_direction)

    ## get object one direction
    x_difference = (entity_coordinates[:, 0] - head_position[:,0]).T
    y_difference = (entity_coordinates[:, 1] - head_position[:,1]).T
    direction = np.array([x_difference, y_difference]).T
    direction = direction / npalg.norm(direction)

    looking_vector = np.zeros((nose_position.shape[0], 1))
    angle_vector = np.zeros((nose_position.shape[0], 1))

    for i in range(looking_vector.shape[0]):
        angle1 = angle_between(head_direction[i], direction[i])
        angle_vector[i] = angle1
        if angle1 < math.pi / 4:
            looking_vector[i, 0] = 1

    return looking_vector,angle_vector

def proximity_vector(position =None, entity_coordinates = None, radius = None):
    ''''
    Checks distance between position vector and entity coordinates
    :param  postion: 2d X N array containing 2D position of nose
            entity_coordinates: 2d X N array containing 2D position of entity
            radius : threshold for distance

    :return: proximity_vector: binary containing information about whether
            the animal is within a certain distance of the other entity
        '''
    ## compute distance between object and mouse
    ## and define proximity vector
    proximity_vector = np.zeros((position.shape[0], 1))
    for i in range(proximity_vector.shape[0]):
        distance = npalg.norm(position[i,:] - entity_coordinates[i,:])
        if distance < radius:
            proximity_vector[i] = 1
    return proximity_vector
