import numpy as np


# Parameters
param = {}
param['Last_N'] = 3  # The parameter for last N measurements (For N/M logic) default: 3
param['Conf_M'] = 2  # The parameter for track confrimation condition (For N/M logic) default: 2
param['Term_M'] = 30  # The parameter for termination condition default: 30
param['Tent_M'] = 2  # The parameter to make the track tentative default: 2
param['lambda'] = 0.1  # 0.6
param['Con_EN'] = True  # Track confirmation logic enable
param['Position_only'] = True  # Using position only for the association if this parameter is true
param['Use_appearance'] = True  # Using appearance feature for the association if this parameter is true


# Parameters for Kalman Filtering and JPDA
q1 = 2.50  # The standard deviation of the process noise for the dynamic model for JPDA 1.00
qm = 2.00  # The standard deviation of the measurement noise for JPDA 1.50
qm_bb = qm * 25  # The standard deviation of the meausremtn noise of the bounding box qm*25
param['PD'] = 0.95  # Detection Probabilty or the average of true positive rate for detections
param['Beta'] = 3 / (1088 * 608)  # Beta is False detection (clutter) density (Poisson assumption) input W * H
param['Gate'] = 9.488 ** 0.5  # Gate size for gating 5.9915: chi2inv95 dim 2. 9.488 for dim 4
param['Gscr'] = 0.0265  # Gating scaling parameter for change the size of gate adaptively, 0.0265 for the time being.
param['Gnum'] = 2  # Gating candidates number, keep only Gnum of candidates in the gating area, inf for not using this limitation.
param['Vmax'] = 50  # maximum velocity that a target can has (used for initialization only) m/s


# Tracking Model
model = {}
model['T'] = 1  # Temporal sampling rate 1/30, 1 for w/o localisation


# Dynamic model
"""
{hat}x_k|k-1 = F{hat}x_k-1|k-1 -> 8x1 = (8x8)(8x1)
P_k|k-1 = FP_k-1|k-1F^T + Q -> 8x8 = (8x8)(8x8)(8x8) + (8x8)
"""
# The transition matrix for the dynamic model, 8x8
model['F'] = [[1, model['T'], 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], 
              [0, 0, 1, model['T'], 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], 
              [0, 0, 0, 0, 1, model['T'], 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], 
              [0, 0, 0, 0, 0, 0, 1, model['T']], [0, 0, 0, 0, 0, 0, 0, 1]]
# The process covariance matrix for the dynamic model 1, 8x8
model['Q'] = [[model['T'] ** 4 / 4 * q1, model['T'] ** 3 / 2 * q1, 0, 0, 0, 0, 0, 0],
              [model['T'] ** 3 / 2 * q1, model['T'] ** 2 * q1    , 0, 0, 0, 0, 0, 0],
              [0, 0, model['T'] ** 4 / 4 * q1, model['T'] ** 3 / 2 * q1, 0, 0, 0, 0],
              [0, 0, model['T'] ** 3 / 2 * q1, model['T'] ** 2 * q1,     0, 0, 0, 0],
              [0, 0, 0, 0, model['T'] ** 4 / 4 * q1, model['T'] ** 3 / 2 * q1, 0, 0],
              [0, 0, 0, 0, model['T'] ** 3 / 2 * q1, model['T'] ** 2 * q1,     0, 0],
              [0, 0, 0, 0, 0, 0, model['T'] ** 4 / 4 * q1, model['T'] ** 3 / 2 * q1],
              [0, 0, 0, 0, 0, 0, model['T'] ** 3 / 2 * q1, model['T'] ** 2 * q1    ]]

# Measurement model
"""
y_k = z_k - H{hat}x_k|k-1 -> 4x1 = (4x1) - (4x8)(8x1)
S = HPH^T + R -> 4x4 = (4x8)(8x8)(8x4) + (4x4)
"""
# The measurement matrix, 4x8
model['H'] = [[1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], 
              [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0]]
# Measurement covariance matrix, 4x4
# model['R'] = qm * np.eye(4)
model['R'] = [[qm, 0, 0, 0], [0, qm, 0, 0], [0, 0, qm_bb, 0],
              [0, 0, 0, qm_bb]]