import numpy as np
import pandas as pd
from numpy import ndarray

def sigmoid(X: ndarray, w: ndarray, b: float) -> float:
    return 1 / (1 + np.exp(-(np.dot(X, w) + b)))


def weight_grad(error, X):
    grad_w = (1 / n) * X.T.dot(error)
    return grad_w

def bias_grad(error):
    grad_b = (1 / n) * np.sum(error)
    return grad_b

def cross_entropy_loss(y, y_roof):
    eps = 1e-15
    y_roof = np.clip(y_roof, eps, 1 - eps)
    return -1 / n * (np.sum((y * np.log(y_roof)) + (1 - y) * np.log(1 - y_roof)))


tests = 5
for test in range(1, tests+1):
    alfa = 0.1
    epochs = 1000

    feat_size = 30

    df = pd.read_csv(f"files/normal_perceptron_data_{test}.csv", sep=',')
    n = df.shape[0]

    X = df.iloc[:, :feat_size].to_numpy()
    y = df["target"].to_numpy()

    w = np.zeros(feat_size)
    b = 0.0



    for epoch in range(epochs):
        y_roof = sigmoid(X, w, b)
        error = y_roof - y
        w = w - alfa * weight_grad(error, X)
        b = b - alfa * bias_grad(error)

        if epoch % 100 == 0:
            loss = cross_entropy_loss(y, y_roof)
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    y_roof = sigmoid(X, w, b)
    y_pred = (y_roof >= 0.5).astype(int)
    accuracy = np.mean(y_pred == y)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    # f = open("files/result_GD.txt", 'a')
    # f.write(f"Accuracy: {accuracy * 100:.2f}%, on test {test}\n")
    # f.close()


