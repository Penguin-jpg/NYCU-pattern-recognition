import typing as t
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import roc_curve, auc


class WeakClassifier(nn.Module):
    """
    Use pyTorch to implement a 1 ~ 2 layer model.
    No non-linear activation allowed.
    """

    def __init__(self, input_dim):
        super(WeakClassifier, self).__init__()
        self.linear1 = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.linear1(x)
        return x


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def tanh(x):
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))


def entropy_loss(outputs, targets):
    # clip the predicted values to avoid log(0)
    outputs = torch.clamp(outputs, min=1e-7, max=1 - 1e-7)

    # since this is a binary classification task, we can use binary cross entropy
    # BCE = -mean(y * log(y_hat) + (1 - y) * log(1 - y_hat))
    loss = -(targets * torch.log(outputs) + (1 - targets) * torch.log(1 - outputs))

    # don't calculate mean here because we have to multiply it the sample weights
    return loss


def plot_learners_roc(
    y_preds: t.List[t.Sequence[float]],
    y_trues: t.Sequence[int],
    fpath="./tmp.png",
):
    fpr, tpr, _ = roc_curve(y_trues, y_preds)
    area = auc(fpr, tpr)

    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot(fpr, tpr, label=f"AUC={area:.4f}")
    plt.legend(loc="lower right")
    plt.savefig(fpath)
    plt.show()


def get_accuracy(y_preds: t.List[t.Sequence[float]], y_trues: t.Sequence[int]):
    num_samples = len(y_trues)
    # round y_preds to get 0 and 1
    num_correct = np.sum(y_trues == np.round(y_preds))
    return num_correct / num_samples


def plot_feature_importance(feature_names, feature_importance):
    plt.title("Feature Importance")
    plt.barh(feature_names, feature_importance)
    plt.gca().invert_yaxis()
    plt.show()
