from qpsolvers import solve_qp

import numpy as np
import matplotlib.pyplot as plt

def SVM(X: np.ndarray, Y: np.ndarray, c: int = 100, epsilon: int = 1) -> tuple:
    m, n = X.shape

    P = np.diag(np.hstack((np.ones(n), np.full(m*2+1, 1e-6))))
    q = np.hstack((np.zeros(n+1), np.full(m*2, c)))

    G = np.vstack((np.hstack((X,                    np.full((m, 1), -1),    np.zeros((m, m)),           np.diag(np.full(m, -1)))),
                   np.hstack((-X,                   np.ones((m, 1)),        np.diag(np.full(m, -1)),    np.zeros((m, m)))),
                   np.hstack((np.zeros_like(X),     np.zeros((m, 1)),       np.zeros((m, m)),           np.diag(np.full(m, -1)))),
                   np.hstack((np.zeros_like(X),     np.zeros((m, 1)),       np.diag(np.full(m, -1)),    np.zeros((m, m))))))
    h = np.hstack((epsilon+Y, epsilon-Y, np.zeros(m*2)))

    A = np.zeros(n+1+m*2)
    b = np.zeros(1)

    res = solve_qp(P, q, G, h, A, b)

    return np.split(res, [n, n+1, n+1+m])

def main():
    X = np.array([
        [1],
        [2],
        [3],
        [4]
    ])
    Y = np.array([5, 10, 12, 15])
    plt.plot(X, Y, "ro")
    
    epsilon = 1
    w, b, z, zs = SVM(X, Y, epsilon=epsilon)
    print(w)
    print(b)
    print(z)
    print(zs)

    x_svm = np.array([X.min(), X.max()])
    plt.plot(x_svm, w*x_svm-b)
    plt.plot(x_svm, w*x_svm-b+epsilon)
    plt.plot(x_svm, w*x_svm-b-epsilon)
    plt.show()

if __name__ == "__main__":
    main()