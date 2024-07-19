# This is the main function which takes the detections, model and
# parameters and generates trajectories.
# ----------------------------------------------------------------------------------------------
# Output:
# Xe: Estimated mean of targets' states representing the trajectories
# - position and velocity at each frame (x_t Vx_t y_t Vy_t)
# Pe: Estimated covariance of the targets' states
# Ff: Frame numbers "t" for which each target's state is estimated.
# Term_Con: Number of consecutive frames which a target is miss-detected before its termination
# Terminated_objects_index: Index of targets that needed to be terminated
# We, He: width and height of the target, not involved in tracking iteration
# Classe: vehicle type, not involved in tracking iteration
# cross_flage: the intersection flag for vehicle crossing the drawn lines, not involved in the
# - tracking iteration
# ----------------------------------------------------------------------------------------------
# Input:
# detection_new: detections, should be in the format of: [[x,y,w,h,class],...,[x,y,w,h,class]]'
# param: The method's parameters
# model: Tracking models
# X0, P0, Term_Con, Terminated_objects_index, W0, H0, Class0, cross_flag: tracking result from last frame
# numlines: number of drawn lines
# ----------------------------------------------------------------------------------------------
import numpy as np
import time
from lib.tracker.JPDA_tracker_fnc import Tree_Constructor, Approx_Multiscan_JPDA_Probabilities

old_JPDA = True


def isnotmember(a, b):
    if a in b:
        return False
    else:
        return True


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


def JPDA(detection_new, cls_feats, model, param, X0, P0, N_cnt, Term_Con, Confirmed_objects_index, Terminated_objects_index, W0, H0, Class0, Track0, Feats0, cdists, Track_idx, frame_tmp): #, cross_flag, numlines):
    # Track termination parameter
    Last_N = param['Last_N']  # The parameter for last N measurements
    Conf_M = param['Conf_M']  # The parameter for track confrimation condition
    Term_M = param['Term_M']  # The parameter for termination condition

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

    DMV = len(H)
    DSV = len(H[0])
    if not (len(X0)):
        N_Target = 0
    else:
        N_Target = len(np.array(X0)[0])

    # ---------------------------------- Parameters from Last Frame -------------------------------------
    # Initial State Vector
    Ff = []
    ij = 0
    for index in list(range(N_Target)):
        Ff.append([1, 1])
        ij += 1
    Pe = P0
    Xe = X0
    Xe = list(np.transpose(Xe))

    # ---------------- loading detections (Measurements) ------------------------------------------------
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
            ccc = [detection_new[2][item] for item in range(len(detection_new[0]))]
            ddd = [detection_new[5][item] for item in range(len(detection_new[0]))]
            Wd = [detection_new[3][item] for item in range(len(detection_new[0]))]
            Hd = [detection_new[4][item] for item in range(len(detection_new[0]))]
            Classd = [detection_new[6][item] for item in range(len(detection_new[0]))]
            ordinaryFrame.append(aaa)   # xi of detection
            ordinaryFrame.append(bbb)   # yi of detection
            ordinaryFrame.append(ccc)   # zi of detection
            ordinaryFrame.append(ddd)   # Af of detection
            ordinaryFrame = np.transpose(ordinaryFrame)

            ordinaryFrame_w.append(Wd)
            ordinaryFrame_h.append(Hd)
            ordinaryFrame_class.append(Classd)
            ordinaryFrame_w = np.transpose(ordinaryFrame_w)
            ordinaryFrame_h = np.transpose(ordinaryFrame_h)
            ordinaryFrame_class = np.transpose(ordinaryFrame_class)

    length_mes = len(ordinaryFrame)
    MXe = np.zeros((DSV, N_Target)) # Expanding the dimension of measurement matirx for N_Targets
    PXe = np.zeros((DSV, DSV, N_Target))
    S = np.zeros((DMV, DMV, N_Target))
    K = np.zeros((DSV, DMV, N_Target)) 
    Target_Obs_indx = [[] for item in range(N_Target)]
    Target_probabilty = [[] for item in range(N_Target)]
    Curr_Obs = []
    Xe = np.array(Xe)
    Pe = np.array(Pe)

    print(" \n next Frame")
    # ----------------- Kalman Preditction & probability map construction (PDA) ------------------------
    for no in list(range(N_Target)):
        if not (no in Terminated_objects_index):
            ordinaryFrame_corr_tar_index = []
            # (Explain) Target_probability = [TMprob, aaaa] == [constant value, probability]
            Target_Obs_indx[no], Target_probabilty[no], MXe[:, no], PXe[:, :, no], S[:, :, no], K[:, :, no], dis = \
                Tree_Constructor(Xe[no].reshape(7), Pe[no], F, Q, H, R, ordinaryFrame,
                                 ordinaryFrame_corr_tar_index, Gate, PD, Beta, DMV, old_JPDA, no, cdists)
    #     print(dis)
    # print(cdists)

    # time.sleep(3)

    # ---------------- Joint Probabilistic Data Association --------------------------------------------
    Temp_probability = [[] for count in range(N_Target)]
    exist_ind = [isnotmember(item_temp, Terminated_objects_index) for item_temp in range(N_Target)]

    for item_temp in range(N_Target):
        if exist_ind[item_temp]:
            Temp_probability[item_temp] = Target_probabilty[item_temp]
    
    Final_probabilty = Approx_Multiscan_JPDA_Probabilities(Target_Obs_indx, Temp_probability, length_mes)

    # ---------------------------- Update step ---------------------------------------------------------
    We = np.zeros((N_Target, 1))
    He = np.zeros((N_Target, 1))
    Classe = np.zeros((N_Target, 1))
    Tracke = np.zeros((N_Target, 1))
    Featse = np.zeros((N_Target, 128))

    for no in range(N_Target):
        if not (no in Terminated_objects_index):
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
                Term_Con[no] += 1
                We[no][0] = W0[no][0]
                He[no][0] = H0[no][0]
                Classe[no][0] = Class0[no][0]
                Tracke[no][0] = Track0[no][0]
                Featse[no]    = Feats0[no] ######################################################################################## Check
            else:
                N_cnt[no][1] = 1
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
                # Classe[no][0] = np.dot(P_temp[1:], np.transpose(ClassD)) / sum_p
                Classe[no][0] = ClassD[0]
                Tracke[no][0] = Track0[no][0]
                Featse[no]    = Feats0[no] ######################################################################################## Check
                    
            if (frame_tmp[no] < Last_N) and (N_cnt[no].count(1) == Conf_M):
                if no in Confirmed_objects_index:
                    pass
                elif Confirmed_objects_index:
                    Confirmed_objects_index.append(no)
                else:
                    Confirmed_objects_index = [no]

            if (Term_Con[no] <= Term_M) or (Term_Con[no] == 0): # Terminate the track when there more than 5 measurements are missing (6)
                pass
            else:
                if no in Confirmed_objects_index:
                    Confirmed_objects_index.remove(no)
                if Terminated_objects_index:
                    Terminated_objects_index.append(no)
                else:
                    Terminated_objects_index = [no]

            if (frame_tmp[no] >= Last_N) and (N_cnt[no].count(1) < Conf_M) and no not in Confirmed_objects_index:
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

    if New_Targets:
        for ij in range(len(New_Targets)):
            N_cnt.append(list(np.zeros(Last_N, dtype=int)))
            N_cnt[ij][0] = 1
            Term_Con.append(0)

            Xe = np.vstack((Xe, np.dot(np.array(H).T, np.array(ordinaryFrame[New_Targets[ij], :]).T)))  # new target
            Pe = np.vstack((Pe, np.array([P1])))
            
            We = np.vstack((We, np.array(ordinaryFrame_w[New_Targets[ij]])))
            He = np.vstack((He, np.array(ordinaryFrame_h[New_Targets[ij]])))
            Classe = np.vstack((Classe, np.array(ordinaryFrame_class[New_Targets[ij]])))
            Tracke = np.vstack((Tracke, Track_idx + ij))
            Featse = np.vstack((Featse, cls_feats[ij])) ######################################################################################## Check
            
            # np.array(Track_tmp.append(Track_idx + ij))
            frame_tmp = np.append(frame_tmp, 0)

        Track_idx = Track_idx + len(New_Targets)
    # aaaa = np.array([np.zeros(numlines) for item in range(len(New_Targets))])
    # # set the length of cross_flag equal to the (no_of_targets, numlines)
    # if len(aaaa):
    #     cross_flage = np.vstack((cross_flag, np.array([np.zeros(numlines) for item in range(len(New_Targets))])))
    # else:
    #     cross_flage = cross_flag
    return Xe.T, Pe, N_cnt, Term_Con, Confirmed_objects_index, Terminated_objects_index, We, He, Classe, Tracke, Featse, Track_idx, frame_tmp #, cross_flage
