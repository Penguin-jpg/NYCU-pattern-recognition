import typing as t
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .utils import WeakClassifier, sigmoid, tanh, entropy_loss, get_accuracy


class AdaBoostClassifier:
    def __init__(self, input_dim: int, num_learners: int = 10) -> None:
        self.sample_weights = None
        # create 10 learners, dont change.
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(num_learners)
        ]
        self.alphas = []

    def fit(
        self, X_train, y_train, num_epochs: int = 500, learning_rate: float = 0.001
    ):
        """Implement your code here"""
        # initialize sample weights with uniform distribution
        self.sample_weights = torch.ones([X_train.shape[0], 1]) / X_train.shape[0]

        # create optimizer
        optimizers = [
            optim.Adam(learner.parameters(), lr=learning_rate)
            for learner in self.learners
        ]

        # convert X_train and y_train to torch tensors
        X_train = torch.from_numpy(X_train).to(dtype=torch.float32)
        y_train = torch.from_numpy(y_train).unsqueeze(1).to(dtype=torch.float32)

        losses_of_models = []
        for i, (model, optimizer) in enumerate(zip(self.learners, optimizers)):
            total_loss = 0
            # train each model
            for epoch in range(num_epochs):
                predicted = model(X_train)
                # use sigmoid to transform the predictions into probability
                predicted = sigmoid(predicted)

                # calculate loss of the current model
                # remember to multiply the loss with the sample weights
                loss = (entropy_loss(predicted, y_train) * self.sample_weights).mean()
                # loss = entropy_loss(predicted, y_train).mean()
                if epoch % 100 == 0:
                    print(f"model {i}, epoch {epoch}, loss: {loss.item()}")

                # backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            losses_of_models.append(total_loss / num_epochs)

            # find where the predictions are different from the actual labels
            predicted = torch.round(sigmoid(model(X_train))).detach()
            print(f"accuracy {i}: {get_accuracy(predicted.numpy(), y_train.numpy())}")
            wrong = torch.where(predicted != y_train, 1.0, 0.0)
            # print(f"wrong: {wrong.shape}")
            # print(f"sample weights: {self.sample_weights.shape}")

            # calculate the weighted error of the current model
            # weighted_error = (self.sample_weights.T @ wrong) / torch.sum(
            #     self.sample_weights
            # )
            weighted_error = self.sample_weights.T @ wrong
            # make sure that weight error is less or equal to 0.5
            if weighted_error > 0.5:
                weighted_error = 1 - weighted_error

            # append weak learner weights of current epoch
            alpha = (0.5 * torch.log((1 - weighted_error) / weighted_error)).view(1, 1)
            self.alphas.append(alpha)

            # if the prediction is correct, yh(x) = 1, else -1
            y_h = torch.where(predicted == y_train, 1.0, -1.0)
            # print(f"alpha: {alpha.shape}")
            # print(f"y_h :{y_h.shape}")
            # print(torch.exp(-alpha * y_h).shape)
            # update sample weights for next epoch
            self.sample_weights *= torch.exp(alpha * y_h)
            # normalize
            self.sample_weights /= torch.sum(self.sample_weights)

        # convert alphas to torch tensor
        # self.alphas = torch.stack(self.alphas, dim=0).squeeze(2)

        return losses_of_models

    def predict_learners(self, X) -> t.Union[t.Sequence[int], t.Sequence[float]]:
        """Implement your code here"""
        with torch.no_grad():
            # convert to torch tensor
            X = torch.from_numpy(X).to(dtype=torch.float32)
            # use tanh to transform the predictions to range of (-1, 1)
            weak_predicitons = [
                alpha * torch.round(tanh(model(X)))
                for alpha, model in zip(self.alphas, self.learners)
            ]
            weak_predicitons = torch.stack(weak_predicitons, dim=1).squeeze(2)
            # calculate weighted sum of predictions
            weak_predicitons = torch.sum(weak_predicitons, dim=1)
            # print(weak_predictions.shape, self.alphas.shape)

            # use sigmoid to transform the predictions to probability
            preditced_probs = sigmoid(weak_predicitons)

            # weighted sum to get the strong prediction
            # strong_prediction = weak_predictions.T @ self.alphas
            strong_predicitions = torch.where(
                torch.sign(weak_predicitons) > 0, 1.0, 0.0
            )

        return strong_predicitions.numpy(), preditced_probs.numpy()

    def compute_feature_importance(self) -> t.Sequence[float]:
        """Implement your code here"""
        # weight sum of all models' weights
        return torch.sum(
            torch.stack(
                [
                    alpha * model.linear1.weight.detach()
                    for alpha, model in zip(self.alphas, self.learners)
                ],
                dim=0,
            )
            .squeeze(1)
            .permute(1, 0),
            dim=1,
        ).numpy()
        # return torch.sum(
        #     torch.stack(
        #         [model.linear1.weight.detach() for model in self.learners],
        #         dim=0,
        #     )
        #     .squeeze(1)
        #     .permute(1, 0),
        #     dim=1,
        # ).numpy()
