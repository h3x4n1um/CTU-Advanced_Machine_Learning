from qpsolvers import solve_qp

import numpy as np
import matplotlib.pyplot as plt

def main():
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

    P = np.diag(np.append(np.ones(n), 1e-6))
    print(P)

    q = np.zeros(n+1)
    print(q)

    G = np.hstack((np.diag(-Y).dot(X),
                    Y.reshape(-1, 1)))
    print(G)

    h = np.full(m, -1)
    print(h)

    A = np.zeros(n+1)
    print(A)

    b = np.zeros(1)
    print(b)

    ans = solve_qp(P, q, G, h, A, b)
    print(ans)

    x_svm = np.array([X[:, 0].min(), X[:, 0].max()])

    plt.plot(x_svm, (-ans[0]*x_svm+ans[2])/ans[1])
    plt.plot(x_svm, (-ans[0]*x_svm+ans[2]+1)/ans[1])
    plt.plot(x_svm, (-ans[0]*x_svm+ans[2]-1)/ans[1])
    plt.show()

if __name__ == "__main__":
    main()