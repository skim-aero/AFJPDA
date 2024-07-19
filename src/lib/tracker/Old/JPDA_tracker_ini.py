import numpy as np
import scipy.spatial.distance


# Initialisation function for the state x of mulitple targets
# state includes x, y postion, velocity and appearance feature (cosine distance).
# Velocity will be calculated in this function from first and second frames.
def initialisation(path1, path2, param, model):
    # 0bx 1by 2xp 3yp 4ht 5wd 6sc 7xi 8yi 9xw 10yw
    DSE = 2
    XYZ = [[[], [], []], [[], [], []]]  # X, Y, Z
    AF = [[], []]   # Appearance feature
    detections_one = path1
    detections_two = path2
    if len(detections_one) and len(detections_two):
        # first frame
        XYZ[0][0] = [i for i in detections_one[0]]  # xi of detected targets (first frame)
        XYZ[0][1] = [i for i in detections_one[1]]  # yi of "
        XYZ[0][2] = [i for i in detections_one[2]]  # zi of "
        AF[0] = [i for i in detections_one[5]]      # Apprearance feature
        # second frame
        XYZ[1][0] = [i for i in detections_two[0]]  # xi of second frame
        XYZ[1][1] = [i for i in detections_two[1]]  # yi of second frame
        XYZ[1][2] = [i for i in detections_two[2]]  # yi of second frame
        AF[1] = [i for i in detections_two[5]]      # Apprearance feature
        T = model['T']  # Temporal sampling rate to calculate the velocity
        Vmax = param['Vmax']  # Maximum velocity that target can have
        aaa = model['H']  # Measurement matrix
        DMV = len(aaa) - 1  # length of measurement matrix
        X0 = np.zeros((len(XYZ[0][0]), len(model['F'])))  # State matrix
        i = 0
        
        # j = 0
        for item in list(range(DSE)):
            if len(XYZ[0]) != 0:
                if i == 0:
                    X0[:, 6] = AF[0]
                    j = 0
                    for it in list(range(DMV)):
                        X0[:, j * DSE + i] = XYZ[i][j]
                        j += 1
                else:

                    temp = scipy.spatial.distance.cdist(np.transpose(np.array(XYZ[i])), np.transpose(np.array(XYZ[0])))

                    mini = np.argmin(temp, 0)
                    XYZ2M = np.transpose(np.array(XYZ[i]))[mini]
                    k = 0

                    for jtem in list(range(DMV)):
                        XYZ2N = X0[:, k * DSE]

                        X0[:, k * DSE + i] = (XYZ2M[:, k] - XYZ2N) / T
                        X0[:, k * DSE + i] = (abs(X0[:, k * DSE + i]) < Vmax) * X0[:, k * DSE + i]
                        k += 1

                # print('X0', X0)
            else:
                if i == 1:
                    X0 = []
                    print('Nothing detected in the first frame')
            i += 1
    else:
        X0 = []

    return X0


def ismember(a, b):
    if a in b:
        return 0
    else:
        return 1


def cellfundevidedsum(b):
    if type(b) == float:
        c = 1
    else:
        c = [item / sum(b) for item in b]
    return c


def JPDA_ini(detections_one, model, param): #, numlines):
    # cross_flag = [[0 for j in range(numlines)] for i in detections_one[6]]
    Last_N = param['Last_N']  # The parameter for last N measurements

    W1 = [i for i in detections_one[3]]  # wi, bounding box width
    H1 = [i for i in detections_one[4]]  # hi, bounding box height
    X0 = model['X0']  # The initial mean (state)
    P0 = model['P0']  # The initial covariance matrix P

    N_Target = len(X0)  # No. of targets
    Class = [i for i in detections_one[6]]
    Track = [i + 1 for i in list(range(N_Target))]

    Track_idx = N_Target + 1

    # ---------------------------------- Initial Parameters -------------------------------------
    # Initial State Vector

    N_cnt = []  # Frame count for Last_N
    Term_Con = [] # Terminatin condition

    Xe = [] # State
    Pe = [] # Covariance

    We = [] # BB width
    He = [] # BB Height
    Classe = [] # Class
    Tracke = [] # Track

    ij = 0
    #
    for index in list(range(N_Target)):
        N_cnt.append(list(np.zeros(Last_N, dtype=int)))
        N_cnt[index][0] = 1
        Term_Con.append(0)

        Xe.append(list(np.ndarray.flatten(np.array(X0[ij]))))
        Pe.append(P0)

        We.append(list(np.ndarray.flatten(np.array(W1[ij]))))
        He.append(list(np.ndarray.flatten(np.array(H1[ij]))))
        Classe.append(list(np.ndarray.flatten(np.array(int(Class[ij])))))
        Tracke.append(list(np.ndarray.flatten(np.array(int(Track[ij])))))

        ij += 1

    Xe = list(np.transpose(Xe))

    return Xe, Pe, N_cnt, Term_Con, We, He, Classe, Tracke, Track_idx #, cross_flag

