import matplotlib.pyplot as plt
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
        # W=(X^T X)^{-1} X^T Y (W includes bias term)

        # note that we need bias term, so we add a column to X
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        pseudo_inverse = np.linalg.pinv(X)  # (X^T X)^{-1} X^T
        W = np.dot(pseudo_inverse, y)

        # W[0] (w0) is the intercept and W[1:] (w1~w4) is the weights
        self.intercept = W[0]
        self.weights = W[1:]

    def predict(self, X):
        # Y=XW+B
        return np.dot(X, self.weights) + self.intercept


class LinearRegressionGradientdescent(LinearRegressionBase):
    def fit(self, X, y, learning_rate: float = 0.001, epochs: int = 1000, use_L1_regularization: bool = True):
        # initalize weights and intercept
        self.weights = np.random.randn(X.shape[1])
        self.intercept = np.zeros((1,))[0]  # cannot use ndarray or there will be an error
        self.regularization_coef = 0.001  #
        # store losses
        losses = []

        for epoch in range(epochs):
            y_hat = self.predict(X)
            print(y_hat)

            # calculate derivative of loss function w.r.t weights (dL/dW)
            # and derivative of loss function w.r.t intercept (dL/dB):
            # L=MSE+L1_reg=1/N*sum((Y - Y_hat)**2)=1/N*sum((Y - (XW+B))**2)+lambda*sum(abs(W))
            # dL/dW=2/N*sum((Y-(XW+B))*(-X))+lambda*sign(W)=-2/N*sum((Y-(XW+B))*X)+lambda*sign(W)
            # dL/dB=2/N*sum((Y-(XW+B))*(-1))=-2/N*sum((Y-(XW+B)))

            if use_L1_regularization:
                d_L_d_W = np.zeros((X.shape[1],))
                # determine plus or minus based on sign of weights
                for i in range(X.shape[1]):
                    if self.weights[i] > 0:
                        d_L_d_W[i] = -2 / X.shape[0] * np.sum((y - y_hat) * X[:, i]) + self.regularization_coef
                    else:
                        d_L_d_W[i] = -2 / X.shape[0] * np.sum((y - y_hat) * X[:, i]) - self.regularization_coef
            else:
                d_L_d_W = -2 / X.shape[0] * np.sum((y - y_hat) * X.T)

            d_L_d_B = -2 / X.shape[0] * np.sum(y - y_hat)

            loss = compute_mse(y_hat, y)
            if use_L1_regularization:
                loss += self.L1_regularize(self.regularization_coef)

            # update weights and intercept
            self.weights -= learning_rate * d_L_d_W
            self.intercept -= learning_rate * d_L_d_B

            losses.append(loss)

            if epoch % 10000 == 0:
                logger.info(f"EPOCH {epoch}, {loss=}")

        return losses

    def predict(self, X):
        # return np.dot(sigmoid(X), self.weights) + self.intercept
        return np.dot(X, self.weights) + self.intercept

    def plot_learning_curve(self, losses):
        plt.title("Training loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.plot(range(len(losses)), losses, label="Train MSE Loss")
        plt.legend(loc="upper right")
        plt.show()

    def L1_regularize(self, regularization_coef):
        # lambda * np.sum(|w0|+|w1|+...)
        return regularization_coef * np.sum(np.abs(self.weights))


def compute_mse(prediction, ground_truth):
    return np.mean(np.square(prediction - ground_truth))


def main():
    train_df = pd.read_csv("./train.csv")
    train_x = train_df.drop(["Performance Index"], axis=1).to_numpy()
    train_y = train_df["Performance Index"].to_numpy()

    LR_CF = LinearRegressionCloseform()
    LR_CF.fit(train_x, train_y)
    logger.info(f"{LR_CF.weights=}, {LR_CF.intercept=:.4f}")

    LR_GD = LinearRegressionGradientdescent()
    losses = LR_GD.fit(train_x, train_y, learning_rate=1e-4, epochs=750000)
    LR_GD.plot_learning_curve(losses)
    logger.info(f"{LR_GD.weights=}, {LR_GD.intercept=:.4f}")

    test_df = pd.read_csv("./test.csv")
    test_x = test_df.drop(["Performance Index"], axis=1).to_numpy()
    test_y = test_df["Performance Index"].to_numpy()

    y_preds_cf = LR_CF.predict(test_x)
    y_preds_gd = LR_GD.predict(test_x)
    y_preds_diff = np.abs(y_preds_gd - y_preds_cf).sum()
    logger.info(f"Prediction difference: {y_preds_diff:.4f}")

    mse_cf = compute_mse(y_preds_cf, test_y)
    mse_gd = compute_mse(y_preds_gd, test_y)
    diff = ((mse_gd - mse_cf) / mse_cf) * 100
    logger.info(f"{mse_cf=:.4f}, {mse_gd=:.4f}. Difference: {diff:.3f}%")


if __name__ == "__main__":
    main()
