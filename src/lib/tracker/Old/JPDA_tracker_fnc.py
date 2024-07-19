import numpy as np
import networkx as nx
import math
import time
import copy
from timeit import default_timer as timer

    
#  this function is used to perform Kalman filter predict step and construct the probability map for JPDA
def Tree_Constructor(X, P, F, Q, H, R, Z, Z_index, d_G, pD, Beta, M, old_JPDA, no, cdists, lambda_=0.98):
    # Kalman predict
    X_k1_k1 = X
    P_k1_k1 = P
    X_k_k1 = np.dot(np.mat(F), np.mat(np.array(X_k1_k1).reshape(7, 1)))
    P_k_k1 = Q + np.dot(np.dot(np.array(F), np.array(P_k1_k1)), np.array(np.transpose(F)))
    S = np.dot(np.dot(np.mat(H), np.mat(P_k_k1)), np.mat(np.transpose(H))) + R
    K = np.dot((np.dot(np.mat(P_k_k1), np.mat(np.transpose(H)))), np.mat(S).I)

    # Probability map is calculated by the distance between predicted states and measurements
    if len(Z) == 0:
        TMindx = []
    else:
        num_meas = 0
        # we use Mahalanobis distance to calculate the probabilities
        mahalanobisdis = []
        # Calculate Mahalanobis distance between one measurement to one track (and move to next measurement)
        # Outer loop (in JPDA_tracker) calculates for entire tracks

        for index in Z:
            m_middle = (np.dot((Z[num_meas] - np.transpose(np.dot(H, X_k_k1))), S.I))
            m_final = math.sqrt(np.dot(m_middle, (np.transpose(Z[num_meas] - np.transpose(np.dot(H, X_k_k1))))))

            # if m_final > d_G:
            #     m_final = np.inf

            # m_final = lambda_ * cdists[no][num_meas] + (1 - lambda_) * m_final

            mahalanobisdis.append(m_final)
            num_meas += 1

        # time.sleep(10)
        # num_meas = 0

        # For normalisation
        # maxdist = max(mahalanobisdis)
        # mindist = min(mahalanobisdis)

        # Calculate the fused distance (Cosine distance + mahalanobis distance)
        # for index in Z:
        #     normdist = (mahalanobisdis[num_meas] - mindist)/(maxdist - mindist)
        #     fusedist = lambda_ * cdists[no][num_meas] + (1 - lambda_) * normdist

        #     print(fusedist)

        #     mahalanobisdis[num_meas] = fusedist
        #     num_meas += 1

        dis = sorted(mahalanobisdis)  # ranking from small to large distance
        inx = np.argsort(mahalanobisdis)    # Return indicies by distance

        # Gating, returns index of measurements within the gating area
        if old_JPDA:
            TMindx = [i for i, j in zip(inx, dis) if j < d_G]
        else:
            # ranked index,from mahalanobisdis closer to further + Gating
            TMindx = [int(Z_index[i]) for i, j in zip(inx, dis) if j < d_G]
    if len(TMindx) == 0:    # If no measurement within gating area aaaa returns TMprob
        aaaa = [(1 - pD) * Beta]
    else:
        DS = [i for i in dis if i < d_G]
        aaa = (((2 * 3.1415926533) ** (M / 2)) * np.linalg.det(np.array(S)) ** 0.5)
        gij = math.e ** (- np.array(DS) * (np.array(DS) / 2)) / aaa # eq. 12 in YOLO + JPDA paper
        TMprob = (1 - pD) * Beta
        aaaa = gij * pD
        aaaa = np.insert(aaaa, 0, TMprob)   # aaaa = [TMprob, aaaa] == [constant value, probability]
        TMprob = np.vstack((np.array(TMprob).reshape(-1, 1), (gij * pD).reshape(-1, 1)))
    return TMindx, aaaa, X_k_k1.reshape(7), P_k_k1, S, K, mahalanobisdis


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
    M_hypo = copy.deepcopy(M_hypo_r)
    for item in M_hypo:
        item.insert(0, -1)
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

        while ~((a == Hypo_indx).all()):
            hypothesis = np.zeros(N_T)
            for j in range(N_T - 1, -1, -1):
                if a[j] > Hypo_indx[j]:
                    a[j] = 0
                    a[j - 1] += 1
                PT[0][j] = M_prob[j][a[j]]
                hypothesis[j] = M_hypo[j][a[j]]
            chkk = 0
            zhpo = np.where(hypothesis == -1)
            if ((zhpo == []) and (len(zhpo[0]) == N_T)) or (len(np.unique(hypothesis)) == N_T - len(zhpo[0]) + 1):
                chkk += 1

            if chkk == 1:
                for i in range(N_T):
                    indd = [item for item in range(len(M_hypo[i])) if M_hypo[i][item] == M_hypo[i][a[i]]]
                    for itemTemp in indd:
                        F_Pr[i][itemTemp] += np.prod(PT)
            a = a + temp
    F_Pr = np.array(F_Pr, dtype=object)
    for item in range(len(F_Pr)):
        F_Pr[item] = F_Pr[item] / sum(F_Pr[item])   # Normalisation of probabilistic
    return F_Pr