"""
Gaussian elimination with modular multiplicative inverse, could also remove the inverse

the example I programmed it to work for was
AX = C
solve for vector X
where A = [[A11, A12, ...],[A21,A22,...],...] and X = [x1,x2,...], and C = [c1,c2,...]
C contains constants
for example A = [[900000006, 500000004], [1,1]], X = [x1,x2], C = [0, 1]
"""
import numpy as np
import sys
def gaussian_elimination():
    # HARD CODED INPUTS
    A = [[900000006, 500000004], [1,1]]
    N = len(A)
    C = [0,1]
    for i, c in enumerate(C):
        A[i].append(c)
    A = np.array(A)
    X = np.zeros(N, dtype=int)
    MOD = int(1e9)+7

    # GAUSS ELIMINATION
    for i in range(N):
        if A[i][i] == 0:
            sys.exit('Divide by zero detected!')
        for j in range(i+1,N):
            ratio = (A[j][i]*pow(int(A[i][i]),-1,MOD))%MOD
            for k in range(N+1):
                A[j][k] = (A[j][k] - ratio*A[i][k] + MOD)%MOD

    # BACK SUBSTITUTION
    X[N-1] = (A[N-1][N]*pow(int(A[N-1][N-1]),-1,MOD))%MOD

    for i in range(N-2,-1,-1):
        X[i] = A[i][N]
        for j in range(i+1,N):
            X[i] = (X[i] - A[i][j]*X[j] + MOD)%MOD
        X[i] = (X[i]*pow(int(A[i][i]),-1,MOD))%MOD
    print(X)