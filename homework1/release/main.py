import numpy as np
import pandas as pd
from loguru import logger


class LinearRegressionBase:
    def __init__(self):
        self.weights = None
        self.intercept = None

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


class LinearRegressionCloseform(LinearRegressionBase):
    def fit(self, X, y):
        # closed-form solution:
        # - W=(X^T X)^{-1} X^T Y (W includes bias term)

        # note that we need bias term, so we add a column to X
        X = np.hstack((X, np.ones((X.shape[0], 1))))

        pseudo_inverse = np.linalg.pinv(X)  # (X^T X)^{-1} X^T
        W = np.dot(pseudo_inverse, y)

        # W[0] is the bias and W[1:] is the weights
        self.intercept = W[0]
        self.weights = W[1:]

    def predict(self, X):
        # Y=XW+B
        return np.dot(X, self.weights) + self.intercept


class LinearRegressionGradientdescent(LinearRegressionBase):
    def fit(self, X, y, learning_rate: float = 0.001, epochs: int = 1000):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def plot_learning_curve(self):
        raise NotImplementedError


def compute_mse(prediction, ground_truth):
    return 0.5 * np.mean((prediction - ground_truth) ** 2)


def main():
    train_df = pd.read_csv("./train.csv")
    train_x = train_df.drop(["Performance Index"], axis=1).to_numpy()
    train_y = train_df["Performance Index"].to_numpy()

    LR_CF = LinearRegressionCloseform()
    LR_CF.fit(train_x, train_y)
    logger.info(f"{LR_CF.weights=}, {LR_CF.intercept=:.4f}")

    # LR_GD = LinearRegressionGradientdescent()
    # losses = LR_GD.fit(train_x, train_y, learning_rate=1e-2, epochs=1000)
    # LR_GD.plot_learning_curve(losses)
    # logger.info(f'{LR_GD.weights=}, {LR_GD.intercept=:.4f}')

    test_df = pd.read_csv("./test.csv")
    test_x = test_df.drop(["Performance Index"], axis=1).to_numpy()
    test_y = test_df["Performance Index"].to_numpy()

    y_preds_cf = LR_CF.predict(test_x)
    # y_preds_gd = LR_GD.predict(test_x)
    # y_preds_diff = np.abs(y_preds_gd - y_preds_cf).sum()
    # logger.info(f'Prediction difference: {y_preds_diff:.4f}')

    mse_cf = compute_mse(y_preds_cf, test_y)
    # mse_gd = compute_mse(y_preds_gd, test_y)
    # diff = ((mse_gd - mse_cf) / mse_cf) * 100
    # logger.info(f'{mse_cf=:.4f}, {mse_gd=:.4f}. Difference: {diff:.3f}%')
    logger.info(f"{mse_cf=:.4f}")


if __name__ == "__main__":
    main()
