import numpy as np
from itertools import chain, combinations

# a function that checks for inequality constraints
def feasible(U1,U2,s1,s2,p,q,v1,v2):
    for i in range(3):
        if s1[0][i] == 0:
            if p[0][i] != 0: return False
            if np.matmul(U1[i], q[0]) > v1: return False
        else:
            if p[0][i] < 0: return False
        if s2[0][i] == 0:
            if q[0][i] != 0: return False
            if np.matmul(U2.T[i], p[0]) > v2: return False
        else:
            if q[0][i] < 0: return False
    return True

# SEM Algorithm, returns a list of all nash equilibria
def SEM(U1, U2):
    list_NE=[]
    f = [s for s in chain.from_iterable(combinations(range(3), r) for r in range(4)) if len(s) > 0]
    for x in f:
        s1 = np.zeros((1,3))
        for i in x: s1[0][i] = 1
        for y in f:
            s2 = np.zeros((1,3))
            for i in y: s2[0][i] = 1
            ixgrid = np.ix_(list(map(bool,s1[0])), list(map(bool,s2[0])))
            U1_new = U1[ixgrid]
            U2_new = U2[ixgrid]

            matrix1 = np.concatenate((np.concatenate((np.concatenate(((-1) * np.ones((U1_new.shape[0],1)), np.zeros((U1_new.shape[0],1))), axis =1), U1_new), axis = 1), np.zeros((U1_new.shape[0],U1_new.shape[0]))), axis = 1)
            matrix2 = np.concatenate((np.concatenate((np.concatenate((np.zeros((U2_new.T.shape[0],1)), (-1) * np.ones((U2_new.T.shape[0],1))), axis =1), np.zeros((U2_new.T.shape[0],U2_new.T.shape[0]))), axis = 1), U2_new.T), axis = 1)
            matrix3 = np.concatenate((np.concatenate((np.zeros((1,2)),np.ones((1,U1_new.shape[1]))), axis = 1), np.zeros((1,U1_new.shape[0]))), axis = 1)
            matrix4 = np.concatenate((np.concatenate((np.zeros((1,2)),np.zeros((1,U1_new.shape[1]))), axis = 1), np.ones((1,U1_new.shape[0]))), axis = 1)
            matrix = np.concatenate((np.concatenate((np.concatenate((matrix1, matrix2), axis = 0), matrix3), axis = 0), matrix4), axis = 0)
            b = np.concatenate((np.zeros((1,U1_new.shape[0]+U1_new.shape[1])), np.ones((1,2))), axis = 1)
            try:
                result = np.linalg.solve(matrix, b[0])
            except np.linalg.linalg.LinAlgError:
                continue
            v1 = result[0]
            v2 = result[1]
            
            p = np.zeros((1,3))
            q = np.zeros((1,3))
            p[0][np.ix_(list(map(bool,s1[0])))] = result[2+U1_new.shape[0]:]
            q[0][np.ix_(list(map(bool,s2[0])))] = result[2:2+U1_new.shape[0]]
            

            if feasible(U1,U2,s1,s2,p,q,v1,v2):
                list_NE.append((p,q))

    return list_NE



U1 = np.array([[3,  0,  0],
               [1,  3,  -2],
               [2,  4,  -1]])
U2 = np.array([[0,  -5,  -4],
               [-1,  3,  4],
               [4,  1,  8]])

list_NE = SEM(U1,U2)
print(list_NE)