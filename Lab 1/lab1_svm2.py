from sklearn.model_selection import train_test_split
from qpsolvers import solve_qp

import numpy as np
import pandas as pd

def SVM(X: np.ndarray, Y: np.ndarray, c: int = 100) -> tuple:
    m, n = X.shape

    P = np.diag(np.hstack((np.ones(n), np.full(m+1, 1e-6))))
    q = np.hstack((np.zeros(n+1), np.full(m, c)))

    G = np.vstack((np.hstack((np.diag(-Y).dot(X),
                               Y.reshape(-1, 1),
                             -np.identity(m))),
                   np.hstack((np.zeros_like(X),
                              np.zeros_like(Y.reshape(-1, 1)),
                             -np.identity(m)))))
    h = np.hstack((np.full(m, -1), np.zeros(m)))

    A = np.zeros(n+1+m)
    b = np.zeros(1)

    ans = solve_qp(P, q, G, h, A, b)
    return np.split(ans, [n, n+1])

def SVM_fit(X_train: pd.DataFrame, Y_train: pd.Series) -> tuple:
    W = []
    B = []
    Z = []

    labels = Y_train.unique()
    for label in labels:
        Y_train_tmp = Y_train.copy()
        Y_train_tmp[Y_train_tmp != label] = -1
        Y_train_tmp[Y_train_tmp == label] = 1

        w, b, z = SVM(X_train.to_numpy(np.float64), Y_train_tmp.to_numpy(np.float64))

        W.append(w)
        B.append(b)
        Z.append(z)
    return W, B, Z

def SVM_predict(X_test: pd.DataFrame, W: list, B: list) -> np.ndarray:
    res = []

    X_test_np = X_test.to_numpy()
    for cnt in range(len(W)):
        res.append(X_test_np.dot(np.vstack(W[cnt]))-B[cnt])

    return np.hstack(res)

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
    print(df.head())
    print(df.describe())

    X = df.drop(columns="class")
    Y = df["class"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=7777777
    )
    print(len(X_train))
    print(len(X_test))

    W, B, Z = SVM_fit(X_train, Y_train)
    labels = Y_train.unique()

    res = SVM_predict(X_test, W, B)
    print(labels)
    print(res)

    Y_pred = labels[res.argmax(1)]
    stats = pd.DataFrame({
        "Y_test": Y_test,
        "Y_pred": Y_pred,
        "Verdict": Y_test == Y_pred,
    })
    print(stats)
    print(stats["Verdict"].value_counts(True))

if __name__ == "__main__":
    main()
