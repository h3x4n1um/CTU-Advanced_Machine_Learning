from qpsolvers import solve_qp

import numpy as np
import matplotlib.pyplot as plt

def main():
    c = 10

    '''
    X = np.array([[4, -5],
                  [4, -6],
                  [-4, -3],
                  [-4, -5],
                  [4, -4],
                  [-4, 3],
                  [3, 1],
                  [-2, 0],
                  [1, 2],
                  [-1, 2]])
    Y = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])
    '''

    X = np.array([[-5, 2],
                  [-5, -3],
                  [-3, -2],
                  [0, 5],
                  [-1, 3],
                  [1, -2]])
    Y = np.array([1, 1, 1, -1, -1, -1])

    plt.plot(X[Y > 0, 0], X[Y > 0, 1], 'bo')
    plt.plot(X[Y < 0, 0], X[Y < 0, 1], 'rx')

    m, n = X.shape

    P = np.diag(np.hstack((np.ones(n), np.full(m+1, 1e-6))))
    print(P)

    q = np.hstack((np.zeros(n+1), np.full(m, c)))
    print(q)

    G = np.vstack((np.hstack((np.diag(-Y).dot(X),
                               Y.reshape(-1, 1),
                              np.identity(m))),
                   np.hstack((np.zeros_like(X),
                              np.zeros_like(Y.reshape(-1, 1)),
                             -np.identity(m)))))
    print(G)

    h = np.hstack((np.full(m, -1), np.zeros(m)))
    print(h)

    A = np.zeros(n+1+m)
    print(A)

    b = np.zeros(1)
    print(b)

    ans = solve_qp(P, q, G, h, A, b)

    w, b, z = np.split(ans, [n, n+1])
    print(w)
    print(b)
    print(z)

    x_svm = np.array([X[:, 0].min(), X[:, 0].max()])

    plt.plot(x_svm, (-w[0]*x_svm+b)/w[1])
    plt.plot(x_svm, (-w[0]*x_svm+b+1)/w[1])
    plt.plot(x_svm, (-w[0]*x_svm+b-1)/w[1])
    plt.show()

if __name__ == "__main__":
    main()