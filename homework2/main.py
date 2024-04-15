import typing as t

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, learning_rate: float = 1e-4, num_iterations: int = 100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.intercept = None

    def fit(
        self,
        inputs: npt.NDArray[float],
        targets: t.Sequence[int],
    ) -> None:
        """
        Implement your fitting function here.
        The weights and intercept should be kept in self.weights and self.intercept.
        """

        # use zero initialization for consistent training results
        self.weights = np.zeros(inputs.shape[1])
        self.intercept = 0

        for iteration in range(self.num_iterations):
            y_preds, _ = self.predict(inputs)

            # calculate gradient
            d_L_d_W = -2 / inputs.shape[0] * np.dot(inputs.T, targets - y_preds)
            d_L_d_B = -2 / inputs.shape[0] * np.sum(targets - y_preds)

            # update weights and intercept
            self.weights = self.weights - self.learning_rate * d_L_d_W
            self.intercept = self.intercept - self.learning_rate * d_L_d_B

            accuracy = accuracy_score(targets, y_preds)

            # print accuracy
            if iteration % 1000 == 0:
                logger.info(f"accuracy at iteration {iteration}: {accuracy:.4f}")

    def predict(
        self,
        inputs: npt.NDArray[float],
    ) -> t.Tuple[t.Sequence[np.float_], t.Sequence[int]]:
        """
        Implement your prediction function here.
        The return should contains
        1. sample probabilty of being class_1
        2. sample predicted class
        """

        class1_prob = self.sigmoid(np.dot(inputs, self.weights) + self.intercept)
        # if predicted probability of being class 1 is greater than 0.5
        predicted_class = np.array([1 if prob > 0.5 else 0 for prob in class1_prob]).astype(np.int8)
        return class1_prob, predicted_class

    def sigmoid(self, x):
        """
        Implement the sigmoid function.
        """
        return 1 / (1 + np.exp(-x))


class FLD:
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None

    def fit(
        self,
        inputs: npt.NDArray[float],
        targets: t.Sequence[int],
    ) -> None:
        # the goal of FLD is to maximize between-class variance and minimize
        # within-class variance

        self.w = np.random.randn(inputs.shape[1])

        # we need to separate data of class 0 and class 1 first
        class0_data = inputs[np.where(targets == 0)]
        class1_data = inputs[np.where(targets == 1)]

        # get mean of two classes
        self.m0 = np.mean(class0_data, axis=0)
        self.m1 = np.mean(class1_data, axis=0)

        # between-class covariance matrix S_B = (m_1 - m_0)(m_1 - m_0)^T
        # the multiplication is not a dot product, it is an outer product
        # details can be found here: https://en.wikipedia.org/wiki/Outer_product

        # you can do this or simply call np.outer
        # sb = np.zeros((self.m0.shape[0], self.m0.shape[0]))
        # diff = self.m1 - self.m0
        # for i in range(self.m0.shape[0]):
        #     for j in range(self.m0.shape[0]):
        #         sb[i, j] = diff[i] * diff.T[j]
        self.sb = np.outer(self.m1 - self.m0, (self.m1 - self.m0).T)

        # within-class covariance matrix S_W = sum((x_n - m_0)(x_n - m_0)^T) +
        # sum((x_n - m_1)(x_n - m_1)^T)
        self.sw = np.dot((class0_data - self.m0).T, (class0_data - self.m0)) + np.dot(
            (class1_data - self.m1).T, (class1_data - self.m1)
        )

        # solve S_W^{-1} S_B w = lambda w to get the optimal w
        # (solving eigenvalue and eigenvector and select the corresponding w of
        # the biggest eigenvalue)
        eigenvalues, eigenvectors = np.linalg.eig(np.dot(np.linalg.inv(self.sw), self.sb))
        self.w = eigenvectors[:, np.argmax(eigenvalues)]
        # w^T w should be 1
        self.w = self.w / np.linalg.norm(self.w)

        # find slope of the projection line
        # in weight space, the slope is w[1] / w[0] = tan(theta) and w^T w = 1,
        # so cos(theta) = w[0] and sin(theta) = w[1]
        # to transform it to feature space, x_original = x_proj * cos(theta)
        # and y_original = y_proj * sin(theta)
        self.slope = self.w[1] / self.w[0]

    def predict(
        self,
        inputs: npt.NDArray[float],
    ) -> t.Sequence[t.Union[int, bool]]:
        # use nearest neighbor to determine the class
        # accuracy=0.5238
        # projected class centers
        # M0 = np.dot(self.m0, self.w)
        # M1 = np.dot(self.m1, self.w)
        # projected_inputs = np.dot(inputs, self.w)
        # if element is closer to M0, it is class 0
        # if element is closer to M1, it is class 1
        # y_preds = np.array([M0 if abs(p - M0) < abs(p - M1) else M1 for p in projected_inputs]).astype(np.int8)
        # return y_preds

        # use the mean of projected input as threshold
        # accuracy=0.6905
        # projected_input = np.dot(inputs, self.w)
        # self.threshold = np.mean(projected_input)
        # y_preds = np.array([0 if p >= self.threshold else 1 for p in projected_input])
        # return y_preds

        # use the mean of projected means as threshold
        # accuracy=0.7381
        M0 = np.dot(self.m0, self.w)
        M1 = np.dot(self.m1, self.w)
        self.threshold = (M0 + M1) / 2
        y_preds = np.array([0 if p >= self.threshold else 1 for p in np.dot(inputs, self.w)])
        return y_preds

    def plot_projection(self, inputs: npt.NDArray[float]):
        plt.title(f"Projection Line: w={self.slope}, b={self.threshold}")

        # x = np.linspace(-2, 2, 100)
        # y = self.slope * x - self.threshold
        # plt.plot(x, y, c="black")
        plt.axline((0, 0), slope=self.slope, c="black")

        predicitons = self.predict(inputs)

        # draw class 0 points
        class0_points = inputs[predicitons == 0]
        plt.scatter(class0_points[:, 0], class0_points[:, 1], c="blue", label="Class 0")
        # draw class 1 points
        class1_points = inputs[predicitons == 1]
        plt.scatter(class1_points[:, 0], class1_points[:, 1], c="red", label="Class 1")

        projected_inputs = np.dot(inputs, self.w)
        # draw projected class 0 points
        projected_class0_points = projected_inputs[predicitons == 0]
        # plot projected points along the direction of discriminant axis
        plt.scatter(projected_class0_points * self.w[0], projected_class0_points * self.w[1], c="blue")
        plt.scatter(projected_class0_points * self.w[0], projected_class0_points * self.w[1], c="blue")
        # draw projected class 1 points
        projected_class1_points = projected_inputs[predicitons == 1]
        plt.scatter(projected_class1_points * self.w[0], projected_class1_points * self.w[1], c="red")

        # draw projection lines for class 0 points
        for class0_point, projected_class0_point in zip(class0_points, projected_class0_points):
            plt.plot(
                [class0_point[0], projected_class0_point * self.w[0]],
                [class0_point[1], projected_class0_point * self.w[1]],
                c="blue",
                alpha=0.3,
            )

        # draw projection lines for class 1 points
        for class1_point, projected_class1_point in zip(class1_points, projected_class1_points):
            plt.plot(
                [class1_point[0], projected_class1_point * self.w[0]],
                [class1_point[1], projected_class1_point * self.w[1]],
                c="blue",
                alpha=0.3,
            )

        plt.legend(loc="upper left")
        # use this to ensure the aspect ratio is correct
        # https://stackoverflow.com/questions/50158333/how-do-i-enforce-a-square-grid-in-matplotlib
        plt.gca().set_aspect("equal")
        plt.show()


def compute_auc(y_trues, y_preds) -> float:
    from sklearn import metrics

    return metrics.roc_auc_score(y_trues, y_preds)


def accuracy_score(y_trues, y_preds) -> float:
    num_samples = len(y_trues)
    # round y_preds to get 0 and 1
    num_correct = np.sum(y_trues == np.round(y_preds))
    return num_correct / num_samples


def main():
    # Read data
    train_df = pd.read_csv("./train.csv")
    test_df = pd.read_csv("./test.csv")

    # Part1: Logistic Regression
    x_train = train_df.drop(["target"], axis=1).to_numpy()  # (n_samples, n_features)
    y_train = train_df["target"].to_numpy()  # (n_samples, )

    x_test = test_df.drop(["target"], axis=1).to_numpy()
    y_test = test_df["target"].to_numpy()

    # LR = LogisticRegression(
    #     learning_rate=1e-2,  # You can modify the parameters as you want
    #     num_iterations=6000,  # You can modify the parameters as you want
    # )
    # LR.fit(x_train, y_train)
    # y_pred_probs, y_pred_classes = LR.predict(x_test)
    # accuracy = accuracy_score(y_test, y_pred_classes)
    # auc_score = compute_auc(y_test, y_pred_probs)
    # logger.info(f"LR: Weights: {LR.weights[:5]}, Intercep: {LR.intercept}")
    # logger.info(f"LR: Accuracy={accuracy:.4f}, AUC={auc_score:.4f}")

    # Part2: FLD
    cols = ["27", "30"]  # Dont modify
    x_train = train_df[cols].to_numpy()
    y_train = train_df["target"].to_numpy()
    x_test = test_df[cols].to_numpy()
    y_test = test_df["target"].to_numpy()

    FLD_ = FLD()
    FLD_.fit(x_train, y_train)
    y_preds = FLD_.predict(x_test)
    accuracy = accuracy_score(y_test, y_preds)
    logger.info(f"FLD: m0={FLD_.m0}, m1={FLD_.m1}")
    logger.info(f"FLD: \nSw=\n{FLD_.sw}")
    logger.info(f"FLD: \nSb=\n{FLD_.sb}")
    logger.info(f"FLD: \nw=\n{FLD_.w}")
    logger.info(f"FLD: Accuracy={accuracy:.4f}")
    FLD_.plot_projection(x_test)


if __name__ == "__main__":
    main()
