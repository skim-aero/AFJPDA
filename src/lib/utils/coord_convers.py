from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import torch
# import torch.nn as nn
import os
import numpy as np
import cv2
import math

from scipy.spatial.distance import cdist
from numpy import random

# Define camera parameters
tvec = np.array([[ 37158.66047278],
                 [-27671.44490781],
                 [177350.84161183]])  # Output translation vecotr from PnPSolver 3x1 wrong

RotationMatrix = np.array([[0.99789284,  0.05464109, -0.03498911],
                           [0.01727485,  0.29605349,  0.95501514],
                           [0.06254172, -0.9536072,   0.29448574]])  # 3x3 wrong

CameraMatrix = np.array([[960, 0.0, 960],
                         [0.0, 960, 540],
                         [0.0, 0.0, 1.0]])   # 3x3

dist_coeffs = [0.0, -0.0, 0.0, 0.0, 0.0]

coeffs = 49537.96821388  # Coefficient s, Scalar wrong

RotationMatrixinv = np.linalg.inv(RotationMatrix)
CameraMatrixinv = np.linalg.inv(CameraMatrix)

# Scene 1
# X_uav = 18
# Y_uav = 137
# Alt = 20
# yaw_deg = 270 - 360

# Scene 2
# X_uav = 118
# Y_uav = 125
# Alt = 25
# yaw_deg = 270 - 185

# Scene 3
X_uav = -47
Y_uav = 165
Alt = 30
yaw_deg = 270 - 270

f = 980
tilt_deg = 30
tilt = tilt_deg * math.pi / 180
yaw = yaw_deg * math.pi / 180


# Newely defined two functions for localisation (2D -> 3D) and projection (3D -> 2D)
def localisation(centre):
    coords = np.zeros((len(centre), 3))

    # for length in range(len(centre)):
    #     temp = RotationMatrixinv @ (coeffs * CameraMatrixinv @ np.array([[centre[length, 0]], [centre[length, 1]], [1]]) - tvec)

    #     coords[length, :] = np.transpose(temp)/1000

    centre = undistort(centre[:, 0:2], CameraMatrix, dist_coeffs)

    for length in range(len(centre)):
        J = Alt/((f * math.tan(tilt) * (math.cos(tilt) ** 2))/(centre[length, 1] - CameraMatrix[1, 2]) + math.cos(tilt) ** 2)
        
        temp_X = ((centre[length, 0] - CameraMatrix[0, 2])/f)*((Alt - J)/(math.tan(tilt) * math.cos(tilt)) + J * math.sin(tilt)) 
        temp_Y = (Alt - J)/math.tan(tilt)

        rot_X = temp_X * math.cos(yaw) - temp_Y * math.sin(yaw)
        rot_Y = temp_X * math.sin(yaw) + temp_Y * math.cos(yaw)

        # target_X = X_uav + ((centre[length, 0] - CameraMatrix[0, 2])/f)*((Alt - J)/(math.tan(tilt) * math.cos(tilt)) + J * math.sin(tilt)) 
        # target_Y = Y_uav - (Alt - J)/math.tan(tilt)

        target_X = X_uav + rot_X
        target_Y = Y_uav - rot_Y
        target_Z = 0.0

        coords[length, :] = np.transpose(np.array([target_X, target_Y, target_Z]))

    return coords


def projection(coords):
    centre = np.zeros((len(coords), 2))

    # for length in range(len(coords)):
    #     temp = 1/coeffs * CameraMatrix @ (RotationMatrix @ np.array([[coords[length, 0]*1000], [coords[length, 1]*1000], [coords[length, 2]*1000]]) + tvec)
    #     centre[length, :] = np.transpose(np.array([temp[0], temp[1]]))

    for length in range(len(coords)):

        # temp_y = (f * math.tan(tilt) * (math.cos(tilt) ** 2) * (Alt + (coords[length, 1] - Y_uav) * math.tan(tilt)))/ \
        #          (Alt - ((math.cos(tilt) ** 2) * (Alt + (coords[length, 1] - Y_uav) * math.tan(tilt)))) + CameraMatrix[1, 2]
        # J = Alt/((f * math.tan(tilt) * (math.cos(tilt) ** 2))/(temp_y - 540) + math.cos(tilt) ** 2)
        # temp_x = (f * (coords[length, 0] - X_uav))/((Alt - J)/(math.tan(tilt) * math.cos(tilt)) + J * math.sin(tilt)) + CameraMatrix[0, 2]

        rot_X = coords[length, 0] - X_uav
        rot_Y = - coords[length, 1] + Y_uav

        temp_X = rot_X * math.cos(yaw) + rot_Y * math.sin(yaw)
        temp_Y = - rot_X * math.sin(yaw) + rot_Y * math.cos(yaw)

        img_y = (f * math.tan(tilt) * (math.cos(tilt) ** 2) * (Alt - temp_Y * math.tan(tilt)))/ \
                 (Alt - ((math.cos(tilt) ** 2) * (Alt - temp_Y * math.tan(tilt)))) + CameraMatrix[1, 2]
        J = Alt/((f * math.tan(tilt) * (math.cos(tilt) ** 2))/(img_y - 540) + math.cos(tilt) ** 2)
        img_x = (f * temp_X)/((Alt - J)/(math.tan(tilt) * math.cos(tilt)) + J * math.sin(tilt)) + CameraMatrix[0, 2]
        
        centre[length, :] = np.transpose(np.array([int(img_x), int(img_y)]))

    centre = distort(centre, CameraMatrix, dist_coeffs)

    return centre


def undistort(xy, k, distortion, iter_num=3):
    k1, k2, p1, p2, k3 = distortion
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[:2, 2]
    for length in range(len(xy)):
        x, y = xy[length].astype(float)
        x = (x - cx) / fx
        x0 = x
        y = (y - cy) / fy
        y0 = y
        for _ in range(iter_num):
            r2 = x ** 2 + y ** 2
            k_inv = 1 / (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3)
            delta_x = 2 * p1 * x*y + p2 * (r2 + 2 * x**2)
            delta_y = p1 * (r2 + 2 * y**2) + 2 * p2 * x*y
            x = (x0 - delta_x) * k_inv
            y = (y0 - delta_y) * k_inv

            temp_x = x * fx + cx
            temp_y = y * fy + cy
            xy[length, :] = np.transpose(np.array([temp_x, temp_y]))

    return xy


def distort(xy, k, distortion, iter_num=3):
    k1, k2, p1, p2, k3 = distortion
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[:2, 2]
    for length in range(len(xy)):
        x, y = xy[length].astype(float)
        x = (x - cx) / fx
        x0 = x
        y = (y - cy) / fy
        y0 = y
        for _ in range(iter_num):
            r2 = x ** 2 + y ** 2
            k = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
            delta_x = 2 * p1 * x*y + p2 * (r2 + 2 * x**2)
            delta_y = p1 * (r2 + 2 * y**2) + 2 * p2 * x*y
            x = x0 * k + delta_x
            y = y0 * k + delta_y

            temp_x = x * fx + cx
            temp_y = y * fy + cy
            xy[length, :] = np.transpose(np.array([temp_x, temp_y]))

    return xy

# points_2D = np.array([[ 1526.6, 714.95],                  # 1 Tesla
#                       [ 1797.2, 819.81],                  # 2 Citroen
#                       [ 925.89, 580.14],                  # 4 Police
#                       [ 937.13, 478.43],                  # 5 Taxi
#                       [ 850.76, 595.21],                  # 6 Etron
#                       [ 947.82, 180.26],                  # 7 T2
#                       [ 1550.2, 758.28],                  # 9 Impala
#                       [ 195.33, 386.25],                  # 12 Firetruck
#                       [ 902.56, 352.94],                  # 13 T2
#                       [ 974.79, 264.03]                   # 14 T2
#                      ], dtype="float32")

# points_3D = np.array([[  -20.538, 130.086, 0.000],         # 1 Tesla
#                       [  -12.714, 137.133, 0.000],         # 2 Citroen
#                       [  -48.905, 119.442, 0.000],         # 4 Police
#                       [  -48.508, 105.731, 0.000],         # 5 Taxi
#                       [  -52.883, 120.482, 0.000],         # 6 Etron
#                       [  -48.827,  -7.652, 0.000],         # 7 T2
#                       [  -20.868, 133.639, 0.000],         # 9 Impala
#                       [ -109.604,  90.084, 0.000],         # 12 Firetruck
#                       [  -52.079,  81.216, 0.000],         # 13 T2
#                       [  -45.168,  48.768, 0.000],         # 14 T2
#                      ], dtype="float32")


# img = cv2.imread("00025.jpg")
# for i in range(len(points_2D)):
#     temp1 = int(points_2D[i][0])
#     temp2 = int(points_2D[i][1])
#     img = cv2.circle(img, (temp1, temp2), radius=4, color=(0, 0, 255), thickness=-1)

# cv2.imshow('image', img)
# cv2.imwrite("result.jpg", img)
 
# # wait for a key to be pressed to exit
# cv2.waitKey(0)

# coords = localisation(points_2D)
# err = coords -points_3D
# print(err)

# # centres = projection(coords)
# # print('centres:', centres)

# dist = np.zeros(len(points_3D))
# dist_uav = np.zeros(len(points_3D))

# totalsum = 0
# totalsum_uav = 0

# for i in range(len(points_3D)):
    
#     dist_x = points_3D[i][0] - coords[i][0]
#     dist_y = points_3D[i][1] - coords[i][1]

#     dist[i] = math.sqrt(dist_x ** 2 + dist_y ** 2)
#     print('dist:', dist[i])
#     totalsum += dist[i]

# avg = totalsum/len(points_3D)

# for i in range(len(points_3D)):
    
#     dist_x = X_uav - points_3D[i][0]
#     dist_y = Y_uav - points_3D[i][1]
#     dist_z = Alt

#     dist_uav[i] = math.sqrt(dist_x ** 2 + dist_y ** 2 + Alt ** 2)
#     print('dist1:', dist_uav[i])

#     totalsum_uav += dist_uav[i]

# avg_uav = totalsum_uav/len(points_3D)

# print('average:', avg)
# print('avg_uav:', avg_uav)