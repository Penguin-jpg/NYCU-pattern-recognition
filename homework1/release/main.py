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
    def fit(
        self,
        X,
        y,
        learning_rate: float = 0.001,
        epochs: int = 1000,
        batch_size: int = 64,
        use_L1_regularization: bool = False,
    ):
        # initalize weights and intercept
        if y.ndim == 1:
            self.weights = np.random.randn(X.shape[1])
        else:
            self.weights = np.random.randn(X.shape[1], 1)

        self.intercept = 0
        self.regularization_coef = 0.001
        # store losses
        losses = []
        rng = np.random.default_rng(seed=42)

        for epoch in range(epochs):
            # for every epoch, randomly shuffle the dataset
            random_indices = rng.choice(len(X), size=len(X), replace=False)
            # split mini batches
            batches = np.array_split(random_indices, len(X) // batch_size)
            loss = 0

            for batch in batches:
                batched_X = X[batch]
                batched_y = y[batch]

                # calculate derivative of loss function w.r.t weights (dL/dW)
                # and derivative of loss function w.r.t intercept (dL/dB):
                # L=MSE+L1_reg=1/N*sum((Y - Y_hat)**2)=1/N*sum((Y - (XW+B))**2)+lambda*sum(abs(W))
                # dL/dW=2/N*sum((Y-(XW+B))*(-X))+lambda*sign(W)=-2/N*sum((Y-(XW+B))*X)+lambda*sign(W)
                # dL/dB=2/N*sum((Y-(XW+B))*(-1))=-2/N*sum((Y-(XW+B)))

                y_hat = self.predict(batched_X)

                if use_L1_regularization:
                    if batched_y.ndim == 1:
                        d_L_d_W = np.zeros((batched_X.shape[1],))
                    else:
                        d_L_d_W = np.zeros((batched_X.shape[1], 1))
                    # determine plus or minus based on sign of weights
                    for i in range(batched_X.shape[1]):
                        if self.weights[i] > 0:
                            d_L_d_W[i] = (
                                -2 / batched_X.shape[0] * np.dot(batched_X[:, i], batched_y - y_hat)
                                + self.regularization_coef
                            )
                        else:
                            d_L_d_W[i] = (
                                -2 / batched_X.shape[0] * np.dot(batched_X[:, i], batched_y - y_hat)
                                - self.regularization_coef
                            )
                else:
                    d_L_d_W = -2 / batched_X.shape[0] * np.dot(batched_X.T, batched_y - y_hat)

                d_L_d_B = -2 / batched_X.shape[0] * np.sum(batched_y - y_hat)

                loss += compute_mse(y_hat, batched_y)
                if use_L1_regularization:
                    loss += self.L1_regularize(self.regularization_coef)

                # update weights and intercept
                self.weights = self.weights - learning_rate * d_L_d_W
                self.intercept = self.intercept - learning_rate * d_L_d_B

            # average over batches
            loss /= len(batches)
            losses.append(loss)

            if epoch % 10000 == 0:
                logger.info(f"EPOCH {epoch}, {loss=}")

        return losses

    def predict(self, X):
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
    losses = LR_GD.fit(
        train_x,
        train_y,
        learning_rate=1e-4,
        epochs=60000,
        batch_size=200,
        use_L1_regularization=True,
    )
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
