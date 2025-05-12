import numpy as np
import pandas as pd

m = 1_000_000
n = 30
np.random.seed(46)

X = np.random.normal(0, 1, (m, n))

w_true = np.random.randn(n)
b_true = np.random.randn()

z = X @ w_true + b_true
p = 1 / (1 + np.exp(-z))

y = np.random.binomial(1, p)

df = pd.DataFrame(X, columns=[f"x{i}" for i in range(1, n + 1)])
df["target"] = y

df.to_csv("files/normal_perceptron_data_5.csv", index=False)
