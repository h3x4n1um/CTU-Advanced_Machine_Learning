from qpsolvers import solve_qp

import numpy as np
import pandas as pd

def SVM(X: np.ndarray, Y: np.ndarray, c: int = 100) -> tuple:
    m, n = X.shape

    P = np.diag(np.hstack((np.ones(n), np.full(m+1, 1e-6))))
    #print(P)

    q = np.hstack((np.zeros(n+1), np.full(m, c)))
    #print(q)

    G = np.vstack((np.hstack((np.diag(-Y).dot(X),
                               Y.reshape(-1, 1),
                             -np.identity(m))),
                   np.hstack((np.zeros_like(X),
                              np.zeros_like(Y.reshape(-1, 1)),
                             -np.identity(m)))))
    #print(G)

    h = np.hstack((np.full(m, -1), np.zeros(m)))
    #print(h)

    A = np.zeros(n+1+m)
    #print(A)

    b = np.zeros(1)
    #print(b)

    ans = solve_qp(P, q, G, h, A, b)
    #print(ans)

    return np.split(ans, [n, n+1])

def main():
    df = pd.read_csv(
        "iris.data",
        names=[
            "sepal length in cm",
            "sepal width in cm",
            "petal length in cm",
            "petal width in cm",
            "class"
        ]
    )
    X = df.drop(columns="class")
    Y = df["class"]

    Y[Y != "Iris-setosa"] = -1
    Y[Y == "Iris-setosa"] = 1

    w, b, z = SVM(X.to_numpy(), Y.to_numpy(np.float64))
    
    print(w)
    print(b)
    print(z)

if __name__ == "__main__":
    main()
