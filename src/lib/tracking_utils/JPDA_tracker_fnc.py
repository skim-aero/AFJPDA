import numpy as np
import networkx as nx
import multiprocessing as mp
import math
import copy
import time
from timeit import default_timer as timer
from scipy.spatial.distance import cdist

    
#  this function is used to perform Kalman filter predict step and construct the probability map for JPDA
def Tree_Constructor(X_k_k1, P_k_k1, S, K, H, R, Z, d_G, Gscr, G_n, pD, Beta, M, no, cdists, lambda_, Position_only, Use_appearance):
    # Probability map is calculated by the distance between predicted states and measurements
    if len(Z) == 0:
        TMindx = []
    else:
        # we use Mahalanobis distance to calculate the probabilities
        mahalanobisdis = []
        # Calculate Mahalanobis distance between one measurement to one track (and move to next measurement)
        # Outer loop (in JPDA_tracker) calculates for entire tracks

        # Adaptive gating (Temp)
        # Mahalanobis gating size calculation
        H_g = [[0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0]]
        X_wh = np.dot(H_g, X_k_k1)
        Diag = math.sqrt(np.dot(np.transpose(X_wh), X_wh))
        d_Gn = Gscr*Diag

        if d_Gn < d_G: d_Gn = d_G

        temp_it = list(range(len(Z)))

        mahalanobisdis = Mahalanobis_dist(X_k_k1, H, Z, S, d_Gn, temp_it, no, cdists, lambda_, Position_only, Use_appearance)

        dis = sorted(mahalanobisdis)  # ranking from small to large distance
        inx = np.argsort(mahalanobisdis)    # Return indicies by distance

        # Gating, returns index of measurements within the gating area
        TMindx = [i for i, j in zip(inx, dis) if j < d_Gn]
        
        # Keep only n (G_n) candidates in the gating area
        if len(TMindx) > G_n: 
            del TMindx[G_n:len(TMindx)+1]

    if len(TMindx) == 0:    # If no measurement within gating area aaaa returns TMprob
        aaaa = [(1 - pD) * Beta]
    else:
        DS = [i for i in dis if i < d_Gn]

        # Keep only n (G_n) candidates in the gating area
        if len(DS) > G_n: 
            del DS[G_n:len(DS)+1]

        aaa = (((2 * 3.1415926533) ** (M / 2)) * np.linalg.det(np.array(S)) ** 0.5)
        gij = math.e ** (- np.array(DS) * (np.array(DS) / 2)) / aaa # eq. 12 in YOLO + JPDA paper
        TMprob = (1 - pD) * Beta
        aaaa = gij * pD
        aaaa = np.insert(aaaa, 0, TMprob)   # aaaa = [TMprob, aaaa] == [constant value, probability]
        TMprob = np.vstack((np.array(TMprob).reshape(-1, 1), (gij * pD).reshape(-1, 1)))

    return TMindx, aaaa, d_Gn


def Mahalanobis_dist(X_k_k1, H, Z, S, d_Gn, index, no, cdists, lambda_=0.50, Position_only=True, Use_appearance=True):
    item = 2 if Position_only else 4

    S = np.matrix(S)

    # Kalman predict
    innovation = Z[index, 0:item] - np.transpose(np.dot(H, X_k_k1)[0:item])
    m_middle = (np.dot(innovation, S[0:item, 0:item].I))
    m_final = np.dot(m_middle, np.transpose(innovation))
    m_final = np.sqrt(np.diag(m_final))

    m_final[m_final > d_Gn] = np.inf

    if Use_appearance:
        m_final = lambda_ * cdists[no, index] + (1 - lambda_) * m_final

    return m_final


def cosdistcalc(trck_features, detections, metric='cosine'):
    cost_matrix = np.zeros(len(detections), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix

    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    cost_matrix = np.maximum(0.0, cdist(trck_features, det_features, metric))   # Nomalised features

    return cost_matrix, det_features


def Approx_Multiscan_JPDA_Probabilities(Tar_mes_Idx, Temp_probability, length_mes):
    length_tar = len(Tar_mes_Idx)
    parse_matrix = np.zeros([length_mes + length_tar, length_mes + length_tar])

    for i in range(len(Tar_mes_Idx)):
        if Tar_mes_Idx[i]:
            for j in Tar_mes_Idx[i]:
                parse_matrix[j + length_tar][i] = 1
   
    parse_matrix = nx.from_numpy_matrix(parse_matrix, create_using=None)
    ccc = sorted(nx.connected_components(parse_matrix), key=len, reverse=True)  # find weakly connected components
    Final_probabilty = [[] for count in range(length_tar)]

    for item in ccc:
        jj = sorted(list(item))
        if jj and min(jj) < length_tar:
            hypo = [Tar_mes_Idx[iii] for iii in range(len(Tar_mes_Idx)) if iii in jj]
            prob = [Temp_probability[jjj] for jjj in range(len(Temp_probability)) if jjj in jj]
            Fr_temp = JPDA_Probabilty_Calculator(prob, hypo)
            FR_final = 0
            for jtem in jj:
                if jtem < length_tar:
                    Final_probabilty[jtem] = Fr_temp[FR_final]
                FR_final += 1

    return Final_probabilty

def JPDA_Probabilty_Calculator(M_prob, M_hypo_r):

    N_T = len(M_hypo_r)  # number of target in the cluster
    M_hypo = [[-1] + item for item in M_hypo_r]

    F_Pr = [[] for count in range(N_T)]  # initialization of final probability
    PT = np.zeros((1, N_T), dtype=float)
    Hypo_indx = np.asarray([len(item) - 1 for item in M_prob])

    for i in range(N_T):
        ind0 = [item for item in range(len(M_prob[i]))]
        F_Pr[i] = np.zeros(len(ind0), dtype=float)

    if N_T == 1:    # If target has only one associated measurement
        F_Pr[0] = M_prob[0]
    else:           # Else, target has multiple associated measurements
        a = np.zeros(N_T, dtype=int)
        temp = np.zeros(N_T, dtype=int)
        temp[-1] = 1

        ind_range = np.arange(N_T-1, -1, -1)
        while ~((a == Hypo_indx).all()):    # Until a == Hypo_indx
            # print(a)
            hypothesis = np.zeros(N_T)
            for j in ind_range:
                if a[j] > Hypo_indx[j]:
                    a[j] = 0
                    a[j - 1] += 1
                PT[0][j] = M_prob[j][a[j]]
                hypothesis[j] = M_hypo[j][a[j]]

            chkk = 0
            zhpo = np.where(hypothesis == -1)
            
            if ((zhpo == ()) and (len(zhpo[0]) == N_T)) or (len(np.unique(hypothesis)) == N_T - len(zhpo[0]) + 1):
                chkk += 1

            if chkk == 1:
                for i in range(N_T):
                    indd = [item for item in range(len(M_hypo[i])) if M_hypo[i][item] == M_hypo[i][a[i]]]
                    for itemTemp in indd:
                        F_Pr[i][itemTemp] += np.prod(PT)
            a += temp    

    F_Pr = np.array(F_Pr, dtype=object)

    for item in range(len(F_Pr)):
        F_Pr[item] = F_Pr[item] / sum(F_Pr[item])   # Normalisation of probabilistic

    return F_Pr