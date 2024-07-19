# This is the main function which takes the detections, model and
# parameters and generates trajectories.
# ----------------------------------------------------------------------------------------------
# Output:
# Xe: Estimated mean of targets' states representing the trajectories
# - position and velocity at each frame (x_t Vx_t y_t Vy_t w_t Vw_t h_t Vh_t)
# Pe: Estimated covariance of the targets' states
# Term_Con: Number of consecutive frames which a target is miss-detected before its termination
# Confirmed_objects_index: Index of targets that confirmed
# Terminated_objects_index: Index of targets that needed to be terminated
# Tentative_objects_index: Index of targets that sorted as tentative
# We, He: width and height of the target, not involved in tracking iteration
# Classe: vehicle type, not involved in tracking iteration
# Tracke: Track number, not involved in tracking iteration
# Featse: Appearance feature from FairMOT, not involved in tracking iteration 
# - tracking iteration
# ----------------------------------------------------------------------------------------------
# Input:
# detection_new: detections, should be in the format of: [[x,y,w,h,class],...,[x,y,w,h,class]]'
# cls_feats: Appearance feature vector from FairMOT
# model: Tracking models
# param: The method's parameters
# X0, P0, Term_Con, Terminated_objects_index, W0, H0, Class0: tracking result from last frame
# ----------------------------------------------------------------------------------------------
import numpy as np
import time
import math
from lib.tracking_utils.JPDA_tracker_fnc import Tree_Constructor, Approx_Multiscan_JPDA_Probabilities, Mahalanobis_dist


def isnotmember(a, b, c):
    if a not in b and a not in c:
        return True
    else:
        return False


def devidedsum(b):
    if b == []:
        return b
    else:
        if type(b) == float:
            c = np.array(1).reshape(-1, 1)
        else:
            print(b)
            c = b / float(sum(b))
        return c


def JPDA(detection_new, cls_feats, model, param, X0, P0, N_cnt, Term_Con, Tent_Con, Confirmed_objects_index, Terminated_objects_index, Tentative_objects_index, W0, H0, Class0, Track0, Feats0, cdists, Track_idx, frame_tmp, Use_appearance = True): #, cross_flag, numlines):
    # Track termination parameter
    Last_N = param['Last_N']  # The parameter for last N measurements
    Conf_M = param['Conf_M']  # The parameter for track confrimation condition
    Tent_M = param['Tent_M']  # The parameter for put to tentative objects
    Term_M = param['Term_M']  # The parameter for termination condition
    Con_En = param['Con_EN']  # Track confirmation logic enable

    lambda_ = param['lambda']
    Position_only = param['Position_only']
    # Use_appearance = param['Use_appearance']

    # Kalman Models
    F = model['F']  # The transition matrix for the dynamic model
    Q = model['Q']  # The process covariance matrix for the dynamic model
    H = model['H']  # Measurement matrix
    R = model['R']  # Measurement covariance matrix
    P1 = model['P0']

    # JPDA Parameters
    PD = param['PD']  # Detection Probability
    Beta = param['Beta']  # False detection (clutter) likelihood
    Gate = param['Gate']  # Gate size for gating
    Gscr = param['Gscr']  # Gate size scaling parameter
    Gnum = param['Gnum']  # Gating candidates number

    DMV = len(H)
    DSV = len(H[0])
    if not (len(X0)):
        N_Target = 0
    else:
        N_Target = len(np.array(X0)[0])

    # print(cls_feats.shape)

    if Use_appearance == False:
        cls_feats = np.zeros((N_Target, 128))
        Feats0 = np.zeros((N_Target, 128))

    # ---------------------------------- Parameters from Last Frame -------------------------------------
    # Initial State Vector
    Ff = []
    ij = 0
    for index in list(range(N_Target)):
        Ff.append([1, 1])
        ij += 1
    Xe = np.array(list(np.transpose(X0)))
    Pe = np.array(P0)

    # ---------------------------- loading detections (Measurements) ------------------------------------
    f = 1  # previous frame
    if f >= 1:  # previous frame
        ordinaryFrame = []
        ordinaryFrame_w = []
        ordinaryFrame_h = []
        ordinaryFrame_class = []
        if not (len(detection_new)):
            ordinaryFrame = []
            ordinaryFrame_w = []
            ordinaryFrame_h = []
            ordinaryFrame_class = []
        else:
            aaa = [detection_new[0][item] for item in range(len(detection_new[0]))]
            bbb = [detection_new[1][item] for item in range(len(detection_new[0]))]
            # ccc = [detection_new[2][item] for item in range(len(detection_new[0]))]
            ccc = [detection_new[3][item] for item in range(len(detection_new[0]))]
            ddd = [detection_new[4][item] for item in range(len(detection_new[0]))]
            Wd = [detection_new[3][item] for item in range(len(detection_new[0]))]
            Hd = [detection_new[4][item] for item in range(len(detection_new[0]))]
            Classd = [detection_new[5][item] for item in range(len(detection_new[0]))]
            ordinaryFrame.append(aaa)   # xi of detection
            ordinaryFrame.append(bbb)   # yi of detection
            ordinaryFrame.append(ccc)   # wi of detection
            ordinaryFrame.append(ddd)   # hi of detection
            ordinaryFrame = np.transpose(ordinaryFrame)

            ordinaryFrame_w.append(Wd)
            ordinaryFrame_h.append(Hd)
            ordinaryFrame_class.append(Classd)
            ordinaryFrame_w = np.transpose(ordinaryFrame_w)
            ordinaryFrame_h = np.transpose(ordinaryFrame_h)
            ordinaryFrame_class = np.transpose(ordinaryFrame_class)

    MXe = np.zeros((DSV, N_Target)) # Expanding the dimension of measurement matirx for N_Targets
    PXe = np.zeros((DSV, DSV, N_Target))
    S = np.zeros((DMV, DMV, N_Target))
    K = np.zeros((DSV, DMV, N_Target))
    length_mes = len(ordinaryFrame) 
    Target_Obs_indx = [[] for item in range(N_Target)]
    Target_probabilty = [[] for item in range(N_Target)]
    Curr_Obs = []

    # ----------------- Kalman Preditction & probability map construction (PDA) ------------------------
    F_new = np.repeat(np.array(F)[:, :, np.newaxis], N_Target, axis=2).transpose(2, 0, 1) # N_Targetx8x8
    F_net = F_new.transpose(0, 2, 1) # N_Targetx8x8, transpose of F

    H_new = np.repeat(np.array(H)[:, :, np.newaxis], N_Target, axis=2).transpose(2, 0, 1) # N_Targetx8x8
    H_net = H_new.transpose(0, 2, 1) # N_Targetx8x8, transpose of H

    Xe_new = Xe.reshape(N_Target, 8, 1) # N_Targetx8x1

    MXe = np.matmul(F_new, Xe_new)
    PXe = np.matmul(np.matmul(F_new, Pe), F_net) + Q
    S = np.matmul(np.matmul(H_new, PXe), H_net) + R
    K = np.matmul(np.matmul(PXe, H_net), np.linalg.inv(S))

    for no in list(range(N_Target)):
        if no not in Tentative_objects_index and no not in Terminated_objects_index:
            # (Explain) Target_probability = [TMprob, aaaa] == [constant value, probability]
            Target_Obs_indx[no], Target_probabilty[no], d_Gn = \
                Tree_Constructor(MXe[no], PXe[no], S[no], K[no], H, R, ordinaryFrame, 
                                    Gate, Gscr, Gnum, PD, Beta, DMV, no, cdists, lambda_, Position_only, Use_appearance)

    MXe = MXe.transpose(1, 2, 0).reshape(8, N_Target)
    PXe = PXe.transpose(1, 2, 0) # 8x8xN_Target
    S = S.transpose(1, 2, 0) # 8x8xN_Target
    K = K.transpose(1, 2, 0) # 8x8xN_Target

    # ------------------------ Joint Probabilistic Data Association ------------------------------------
    Temp_probability = [[] for count in range(N_Target)]
    exist_ind = [isnotmember(item_temp, Tentative_objects_index, Terminated_objects_index) for item_temp in range(N_Target)]
    
    for item_temp in range(N_Target):
        if exist_ind[item_temp]:
            Temp_probability[item_temp] = Target_probabilty[item_temp]
    
    Final_probabilty = Approx_Multiscan_JPDA_Probabilities(Target_Obs_indx, Temp_probability, length_mes)
    
    # -------------------------------------- Update step -----------------------------------------------
    We = np.zeros((N_Target, 1))
    He = np.zeros((N_Target, 1))
    Classe = np.zeros((N_Target, 1))
    Tracke = np.zeros((N_Target, 1))
    Featse = np.zeros((N_Target, 128))

    for no in range(N_Target):
        if no not in Tentative_objects_index and no not in Terminated_objects_index:
            P_temp = np.transpose(Final_probabilty[no])
            if not Target_Obs_indx[no]:
                Xe[no] = MXe[:, no]
                dP = 0
                Pe[no] = PXe[:, :, no]
            else:
                Yij = ordinaryFrame[Target_Obs_indx[no]] - np.tile(np.transpose(np.dot(H, MXe[:, no])),
                                                                   [np.size(Target_Obs_indx[no]), 1])   # Innovation
                Ye = np.dot(P_temp[1:], Yij)    # Weighted innovation
                Ye = Ye.reshape((1, -1))

                Xe[no] = (MXe[:, no]).T + (np.dot(K[:, :, no], Ye.T)).T # Updated state estimate
                dP = np.dot(
                    np.dot(K[:, :, no], (np.dot(np.tile(P_temp[1:], [DMV, 1]) * Yij.T, Yij) - np.dot(Ye.T, Ye))),
                    K[:, :, no].T)  # Updated error covariance

            Pst = PXe[:, :, no] - np.dot(np.dot(K[:, :, no], S[:, :, no]), K[:, :, no].T)

            if len(P_temp.reshape(-1, 1)) != 1:
                Po = P_temp[0] * PXe[:, :, no] + (1 - P_temp[0]) * Pst
            else:
                Po = P_temp * PXe[:, :, no] + (1 - P_temp) * Pst
            Pe[no] = Po + dP  # Updated error covariance

            # ---------------------------- Initiation & Termination --------------------------------------------
            # ---------------------------------- Termination ---------------------------------------------------
            if np.argmax(P_temp.reshape(-1, 1)) == 0:  # Highest association score = 0, no corresponding measurement
                N_cnt[no][1] = 0    # Detection missing
                Tent_Con[no] += 1
                Term_Con[no] += 1
                We[no][0] = W0[no][0]
                He[no][0] = H0[no][0]
                Classe[no][0] = Class0[no][0]
                Tracke[no][0] = Track0[no][0]
                Featse[no]    = Feats0[no]
            else:
                N_cnt[no][1] = 1
                Tent_Con[no] = 0
                Term_Con[no] = 0

                meas_inx = Target_Obs_indx[no]
                sum_p = sum(P_temp[1:])

                temp_int = np.int64(1)
                if type(meas_inx) == type(temp_int):
                    meas_inx = [meas_inx]

                WD = np.zeros(len(meas_inx))
                HD = np.zeros(len(meas_inx))
                ClassD = np.zeros(len(meas_inx))
                TrackD = np.zeros(len(meas_inx))
                iii = 0
                for item in meas_inx:
                    WD[iii] = Wd[item]
                    HD[iii] = Hd[item]
                    ClassD[iii] = Classd[item]
                    iii += 1

                We[no][0] = np.dot(P_temp[1:], np.transpose(WD)) / sum_p
                He[no][0] = np.dot(P_temp[1:], np.transpose(HD)) / sum_p
                # Classe[no][0] = round(np.dot(P_temp[1:], np.transpose(ClassD)) / sum_p)
                Classe[no][0] = Class0[no][0]
                Tracke[no][0] = Track0[no][0]
                Featse[no]    = Feats0[no]

                if Con_En:
                    if (frame_tmp[no] < Last_N) and (N_cnt[no].count(1) == Conf_M):
                        if no in Confirmed_objects_index:
                            pass
                        elif Confirmed_objects_index:
                            Confirmed_objects_index.append(no)
                            if no in Tentative_objects_index:
                                Tentative_objects_index.remove(no)
                        else:
                            Confirmed_objects_index = [no]                 
                else:
                    if no in Confirmed_objects_index:
                        pass
                    elif Confirmed_objects_index:
                        Confirmed_objects_index.append(no)
                        if no in Tentative_objects_index:
                            Tentative_objects_index.remove(no)
                    else:
                        Confirmed_objects_index = [no] 

            if (Tent_Con[no] <= Tent_M) or (Tent_Con[no] == 0): # Terminate the track when there more than 5 measurements are missing (6)
                pass
            else:
                if no in Confirmed_objects_index:
                    Confirmed_objects_index.remove(no)
                if Tentative_objects_index:
                    Tentative_objects_index.append(no)
                else:
                    Tentative_objects_index = [no]

            if (frame_tmp[no] >= Last_N) and (N_cnt[no].count(1) <= Conf_M) and no not in Confirmed_objects_index:
                frame_tmp[no] = 0
                [0 if x == 1 else x for x in N_cnt]
            
            # N_cnt push for using N/M logic
            for cnt in reversed(range(Last_N)):
                N_cnt[no][cnt] = N_cnt[no][cnt-1]
            
            N_cnt[no][0] = 0

            # ---------------------------------- Initiation ---------------------------------------------------
            # finding effective measurements
            if Target_Obs_indx[no]:
                if Curr_Obs == []:
                    Curr_Obs = Target_Obs_indx[no][0]
                else:
                    Curr_Obs = np.vstack(
                        (np.array(Curr_Obs).reshape(-1, 1), np.array(Target_Obs_indx[no]).reshape(-1, 1)))
   
    All_Obs = len(ordinaryFrame)

    if type(Curr_Obs) != np.ndarray:
        Curr_Obs_1 = [Curr_Obs]
    else:
        Curr_Obs_1 = Curr_Obs

    New_Targets = [item for item in range(All_Obs) if not (item in Curr_Obs_1)]
    temp = []

    """
    MXe = MXe.reshape(1, 8, N_Target).transpose(2, 1, 0) # 28x8x1
    PXe = PXe.transpose(2, 1, 0) # 8x8x28
    S = S.transpose(2, 1, 0) # 8x8x28
    K = K.transpose(2, 1, 0) # 8x8x28
    """
    
    # ------------------------------------ Re-confirmation----------------------------------------------
    for no in range(N_Target):
        if no in Tentative_objects_index and no not in Terminated_objects_index:
            Xe[no] = MXe[:, no]
            dP = 0
            Pe[no] = PXe[:, :, no]

            if New_Targets:
                for ij in range(len(New_Targets)):
                    if New_Targets[ij] in temp:
                        pass
                    else:
                        # Mahalanobis distance calculation
                        X_k_k1 = np.dot(np.mat(F), np.mat(np.array(Xe[no].reshape(8)).reshape(8, 1)))
                        P_k_k1 = Q + np.dot(np.dot(np.array(F), np.array(Pe[no])), np.array(np.transpose(F)))
                        
                        S = np.dot(np.dot(np.mat(H), np.mat(P_k_k1)), np.mat(np.transpose(H))) + R
                        
                        m_final = Mahalanobis_dist(X_k_k1, H, ordinaryFrame, S, Gate, New_Targets[ij], no, cdists, lambda_, Position_only, Use_appearance)

                        # Re-confirmation 
                        if m_final < Gate/3:
                            N_cnt[no][1] = 1
                            Tent_Con[no] = 0
                            Term_Con[no] = 0
                            Xe[no] = np.dot(np.array(H).T, np.array(ordinaryFrame[New_Targets[ij], :]).T)
                            Pe[no] = np.array([P1])

                            We[no][0] = np.array(ordinaryFrame_w[New_Targets[ij]])
                            He[no][0] = np.array(ordinaryFrame_h[New_Targets[ij]])
                            Classe[no][0] = np.array(ordinaryFrame_class[New_Targets[ij]])
                            Tracke[no][0] = Track0[no][0]
                            Featse[no]    = Feats0[no]

                            if no in Tentative_objects_index:
                                Tentative_objects_index.remove(no)
                                Confirmed_objects_index.append(no)

                            if New_Targets[ij] not in temp:
                                temp.append(New_Targets[ij])

        if no in Tentative_objects_index and no not in Terminated_objects_index:
            if (Term_Con[no] <= Term_M) or (Term_Con[no] == 0): # Terminate the track when there more than 5 measurements are missing (6)
                Term_Con[no] += 1
                Tracke[no][0] = Track0[no][0]
                Featse[no]    = Feats0[no]
            else:
                if no in Tentative_objects_index:
                    Tentative_objects_index.remove(no)
                if Terminated_objects_index:
                    Terminated_objects_index.append(no)
                    Tracke[no][0] = 0
                else:
                    Terminated_objects_index = [no]
                    Tracke[no][0] = 0

    for no in range(len(temp)):
        if temp[no] in New_Targets:
            New_Targets.remove(temp[no])

    # -------------------------------------- Initiation ------------------------------------------------
    if New_Targets:
        for ij in range(len(New_Targets)):
            N_cnt.append(list(np.zeros(Last_N, dtype=int)))
            N_cnt[ij][0] = 1
            Tent_Con.append(0)
            Term_Con.append(0)
            Xe = np.vstack((Xe, np.dot(np.array(H).T, np.array(ordinaryFrame[New_Targets[ij], :]).T)))  # new target
            Pe = np.vstack((Pe, np.array([P1])))
            
            We = np.vstack((We, np.array(ordinaryFrame_w[New_Targets[ij]])))
            He = np.vstack((He, np.array(ordinaryFrame_h[New_Targets[ij]])))
            Classe = np.vstack((Classe, np.array(ordinaryFrame_class[New_Targets[ij]])))
            Tracke = np.vstack((Tracke, Track_idx + ij))
            Featse = np.vstack((Featse, cls_feats[ij]))
            
            # np.array(Track_tmp.append(Track_idx + ij))
            frame_tmp = np.append(frame_tmp, 0)

        Track_idx = Track_idx + len(New_Targets)

    return Xe.T, Pe, N_cnt, Term_Con, Tent_Con, Confirmed_objects_index, Terminated_objects_index, Tentative_objects_index, We, He, Classe, Tracke, Featse, Track_idx, frame_tmp #, cross_flage
