from qpsolvers import solve_qp

import numpy as np
import matplotlib.pyplot as plt

def linear_kernel(x_i: np.ndarray, x_j: np.ndarray) -> np.float64:
    return x_i.dot(x_j.T)

def polyminal_kernel(x_i: np.ndarray, x_j: np.ndarray, gama: np.float64 = 1.0, d: np.float64 = 2.0, r: np.float64 = 0.0) -> np.float64:
    return (gama*linear_kernel(x_i, x_j) + r)**d

def RBF_kernel(x_i: np.ndarray, x_j: np.ndarray, gama: np.float64 = 1.0) -> np.float64:
    diff = x_i - x_j
    return np.exp(-gama*linear_kernel(diff, diff))

def main():
    c = 10

    '''
    X = np.array([[2, 2],
                  [3, 1],
                  [1, 1],
                  [2.5, 1.4]])
    Y = np.array([1, 1, -1, -1])
    '''

    X = np.array([[1, 0],
                  [0, 1],
                  [-1, 0],
                  [0, -1],
                  [2, 0],
                  [0, 2],
                  [-2, 0],
                  [0, -2],
                  [0, -3]])
    Y = np.array([1, 1, 1, 1, -1, -1, -1, -1, -1])

    m, n = X.shape

    D = np.diag(Y)

    kernel = linear_kernel
    K = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            K[i][j] = kernel(X[i], X[j])

    P = D.dot(K).dot(D)
    P = 0.5*(P + P.T)
    P = P + np.diag(np.full(m, 1e-10))
    print(P)

    q = -np.ones(m)
    print(q)

    G = np.vstack((-np.identity(m), np.identity(m)))
    print(G)

    h = np.hstack((np.full(m, 0), np.full(m, c)))
    print(h)

    A = Y
    print(A)

    b = np.zeros(1)
    print(b)

    alpha = solve_qp(P, q, G, h, A, b, verbose=True)
    print(alpha)

    for k in range(len(alpha)):
        if alpha[k] < c:
            b = 0
            for i in range(m):
                b = b + alpha[i]*Y[i]*kernel(X[i], X[k])
            break
    print(b)
    
    return alpha, b

if __name__ == "__main__":
    main()