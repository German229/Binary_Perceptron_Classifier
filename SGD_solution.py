import numpy as np
import pandas as pd
from numpy import ndarray

def sigmoid(x_i: ndarray, w: ndarray, b: float) -> float:
    return 1 / (1 + np.exp(- (np.dot(x_i, w) + b)))

def weight_grad(error, x_i):
    return error * x_i

def bias_grad(error):
    return error

def cross_entropy_loss(y, y_roof):
    eps = 1e-15
    y_roof = np.clip(y_roof, eps, 1 - eps)
    return - (y * np.log(y_roof) + (1 - y) * np.log(1 - y_roof))

tests = 2
for test in range(1, tests+1):

    alfa = 0.1
    epochs = 5
    feat_size = 30

    df = pd.read_csv(f"files/normal_perceptron_data_{test}.csv", sep=',')
    n = df.shape[0]
    X = df.iloc[:, :feat_size].to_numpy()
    y = df["target"].to_numpy()

    w = np.zeros(feat_size)
    b = 0.0


    for epoch in range(epochs):
        indices = np.random.permutation(n)
        X = X[indices]
        y = y[indices]

        total_loss = 0

        for i in range(n):
            x_i = X[i]
            y_i = y[i]

            y_roof = sigmoid(x_i, w, b)
            error = y_roof - y_i

            w = w - alfa * weight_grad(error, x_i)
            b = b - alfa * bias_grad(error)

            total_loss += cross_entropy_loss(y_i, y_roof)

        avg_loss = total_loss / n
        print(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}")


    y_roof_all = sigmoid(X, w, b)
    y_pred = (y_roof_all >= 0.5).astype(int)
    accuracy = np.mean(y_pred == y)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    # f = open("files/result_SGD.txt", 'a')
    # f.write(f"Accuracy: {accuracy * 100:.2f}%, on test {test}\n")
    # f.close()
