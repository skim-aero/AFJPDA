import numpy as np


# Parameters
param = {}
param['Last_N'] = 5  # The parameter for last N measurements (For N/M logic) default: 5
param['Conf_M'] = 3  # The parameter for track confrimation condition (For N/M logic) default: 3
param['Term_M'] = 10  # The parameter for termination condition default: 5
param['tret'] = 15


# Parameters for Kalman Filtering and JPDA
q1 = 1.00 # The standard deviation of the process noise for the dynamic model for JPDA 30000 for localisation/1.00 for w/o
qm = 1.50  # The standard deviation of the measurement noise for JPDA
KFq1 = 30000  # The standard deviation of the process noise for the dynamic model for JPDA
KFqm = 1.50  # The standard deviation of the measurement noise for JPDA
param['PD'] = 0.95  # Detection Probabilty or the average of true positive rate for detections
param['Beta'] = 3 / (1088 * 608)  # Beta is False detection (clutter) density (Poisson assumption) input W * H
param['Gate'] = 5.9915 # Gate size for gating ############### 1 ** 0.5 for localisation and 30 ** 0.5 for w/o localisation 5.9915: chi2inv95 dim 2
param['Vmax'] = 50  # maximum velocity that a target can has (used for initialization only) m/s
param['S_limit'] = 100  # parameter to stop the gate size growing too much


# Tracking Model
model = {}
model['T'] = 1  # Temporal sampling rate 1/30, 1 for w/o localisation


# Dynamic model
"""
{hat}x_k|k-1 = F{hat}x_k-1|k-1 -> 5x1 = (5x5)(5x1)
P_k|k-1 = FP_k-1|k-1F^T + Q -> 5x5 = (5x5)(5x5)(5x5) + (5x5)
"""
# The transition matrix for the dynamic model, 5x5
model['F'] = [[1, model['T'], 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, model['T'], 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, model['T'], 0], [0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 1]]  
# The process covariance matrix for the dynamic model 1, 5x5
model['Q'] = [[model['T'] ** 4 / 4 * q1, model['T'] ** 3 / 2 * q1, 0, 0, 0, 0, 0],
              [model['T'] ** 3 / 2 * q1, model['T'] ** 2 * q1, 0, 0, 0, 0, 0],
              [0, 0, model['T'] ** 4 / 4 * q1, model['T'] ** 3 / 2 * q1, 0, 0, 0],
              [0, 0, model['T'] ** 3 / 2 * q1, model['T'] ** 2 * q1, 0, 0, 0],
              [0, 0, 0, 0, model['T'] ** 4 / 4 * q1, model['T'] ** 3 / 2 * q1, 0],
              [0, 0, 0, 0, model['T'] ** 3 / 2 * q1, model['T'] ** 2 * q1, 0],
              [0, 0, 0, 0, 0, 0, model['T'] ** 2 * q1]]
# The process covariance matrix for the dynamic model 1, 5x5
model['KFQ'] = [[model['T'] ** 4 / 4 * q1, model['T'] ** 3 / 2 * q1, 0, 0, 0, 0, 0],
              [model['T'] ** 3 / 2 * q1, model['T'] ** 2 * q1, 0, 0, 0, 0, 0],
              [0, 0, model['T'] ** 4 / 4 * q1, model['T'] ** 3 / 2 * q1, 0, 0, 0],
              [0, 0, model['T'] ** 3 / 2 * q1, model['T'] ** 2 * q1, 0, 0, 0],
              [0, 0, 0, 0, model['T'] ** 4 / 4 * q1, model['T'] ** 3 / 2 * q1, 0],
              [0, 0, 0, 0, model['T'] ** 3 / 2 * q1, model['T'] ** 2 * q1, 0],
              [0, 0, 0, 0, 0, 0, model['T'] ** 2 * q1]]
# model['Q'] = [[0.025, 0.05, 0, 0, 0], [0.05, 0.1, 0, 0, 0],
#               [0, 0, 0.025, 0.05, 0], [0, 0, 0.05, 0.1, 0],
#               [0, 0, 0, 0, 0.1]]


# Measurement model
"""
y_k = z_k - H{hat}x_k|k-1 -> 3x1 = (3x1) - (3x5)(5x1)
S = HPH^T + R -> 3x3 = (3x5)(5x5)(5x3) + (3x3)
"""
# The measurement matrix, 3x5
model['H'] = [[1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], 
              [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1]]
# Measurement covariance matrix, 3x3
model['R'] = qm * np.eye(4)
model['KFR'] = KFqm * np.eye(4)