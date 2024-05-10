import typing as t
import numpy as np
import torch
import torch.optim as optim
from .utils import WeakClassifier, sigmoid, entropy_loss, get_accuracy
import random


class BaggingClassifier:
    def __init__(self, input_dim: int) -> None:
        # don't change these
        random.seed(55)
        np.random.seed(55)
        torch.manual_seed(55)
        # create 10 learners, dont change.
        self.learners = [WeakClassifier(input_dim=input_dim) for _ in range(10)]

    def bootstrap_sampling(self, X_train, y_train):
        # sample with replacement
        random_indices = np.random.choice(
            np.arange(X_train.shape[0]), size=X_train.shape[0], replace=True
        )

        return torch.from_numpy(X_train[random_indices]).to(
            dtype=torch.float32
        ), torch.from_numpy(y_train[random_indices]).unsqueeze(1).to(
            dtype=torch.float32
        )

    def fit(self, X_train, y_train, num_epochs: int, learning_rate: float):
        """Implement your code here"""

        losses_of_models = []
        # create optimizer
        optimizers = [
            optim.Adam(learner.parameters(), lr=learning_rate)
            for learner in self.learners
        ]

        for i, (model, optimizer) in enumerate(zip(self.learners, optimizers)):
            # get corresponding bootstrap dataset
            X, y = self.bootstrap_sampling(X_train, y_train)
            total_loss = 0

            # train each model
            for epoch in range(num_epochs):
                # use sigmoid to transform the predictions into probability
                predicted = sigmoid(model(X))
                # predicted = model(X).sigmoid()

                # calculate loss of the current model
                loss = entropy_loss(predicted, y).mean()
                if epoch % 1000 == 0:
                    print(f"model {i}, epoch {epoch}, loss: {loss.item()}")

                # backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            losses_of_models.append(total_loss / num_epochs)
            print(
                f"accuracy {i}: {get_accuracy(sigmoid(model(X)).detach().numpy(), y.numpy())}"
            )

        return losses_of_models

    def predict_learners(self, X) -> t.Union[t.Sequence[int], t.Sequence[float]]:
        """Implement your code here"""
        X = torch.from_numpy(X).to(dtype=torch.float32)
        # get predicition of every model and do a majority voting
        with torch.no_grad():
            predictions = torch.stack(
                [sigmoid(model(X)) for model in self.learners], dim=1
            ).squeeze(2)

            majority = len(self.learners) // 2

            predicted_classes = [
                1 if len(prediction[prediction > 0.5]) > majority else 0
                for prediction in predictions
            ]

            # use expectation as the predicted probability
            predicted_probs = torch.mean(predictions, dim=1)

        return np.array(predicted_classes), predicted_probs.numpy()

    def compute_feature_importance(self) -> t.Sequence[float]:
        """Implement your code here"""
        return torch.sum(
            torch.stack(
                [
                    # use abs to prevent negative values
                    model.linear1.weight.abs().detach()
                    for model in self.learners
                ],
                dim=0,
            )
            .squeeze(1)
            .permute(1, 0),
            dim=1,
        ).numpy()
