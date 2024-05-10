"""
You dont have to follow the stucture of the sample code.
However, you should checkout if your class/function meet the requirements.
"""

import typing as t
import numpy as np

# implementation modified from https://medium.com/@omidsaghatchian/decision-tree-implementation-from-scratch-visualization-5eb0bbf427c2


class TreeNode:
    def __init__(
        self,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
        predicted_class=None,
        is_leaf=False,
        num_samples=None,
        gini=None,
    ):
        # index of the feature splitted on
        self.feature_index = feature_index
        # threshold for spliiting
        self.threshold = threshold
        # left and right subtrees
        self.left, self.right = left, right
        # predicted class (note that only leaf node has a predicted class)
        self.predicted_class = predicted_class
        # is this node a leaf node
        self.is_leaf = is_leaf
        # to calculate feature importance, we have to store the number of samples and gini index
        self.num_samples = num_samples
        self.gini = gini


class DecisionTree:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth

        # minimum number of samples required to be at a leaf node
        # this can help to prevent overfitting
        self.min_samples = 10

    def fit(self, X, y):
        # store number of samples to calculate feature importance
        self.num_samples = X.shape[0]

        # get potential splits (thresholds)
        self.get_potential_splits(X)

        # generate decision tree
        self.generate_tree(X, y)

    def predict(self, X):
        predicitons = []

        # iterate every row
        for x in X:
            predicitons.append(self._predict_tree(x, self.root))

        predicitons = np.array(predicitons)
        return predicitons

    # traverse the decision tree
    def _predict_tree(self, x, node):
        # print(f"split on: {node.feature_index}")
        # print(f"threshold: {node.threshold}")
        # print(f"is leaf: {node.is_leaf}")
        # print(f"predicted class: {node.predicted_class}\n")

        # if it is leaf node, return the predicted class of the node
        if node.is_leaf:
            return node.predicted_class

        if node.left is not None and x[node.feature_index] <= node.threshold:
            # print("left")
            return self._predict_tree(x, node.left)

        if node.right is not None and x[node.feature_index] > node.threshold:
            # print("right")
            return self._predict_tree(x, node.right)

    def get_potential_splits(self, X):
        # for each feature, compute all thresholds
        self.feature_thresholds = {}
        for feature_index in range(X.shape[1]):
            self.feature_thresholds[feature_index] = []
            # sort the data based on the feature
            sorted_data = np.sort(X[:, feature_index])
            # the mean of i and i+1 is the threshold i in this feature
            for i in range(len(sorted_data) - 1):
                self.feature_thresholds[feature_index].append(
                    (sorted_data[i] + sorted_data[i + 1]) / 2
                )

    # Split dataset based on a feature and threshold
    def split_dataset(self, X, y, feature_index, threshold):
        feature = X[:, feature_index]
        # split into two parts: less than or equal to threshold and greater than threshold
        X_left, y_left = X[feature <= threshold], y[feature <= threshold]
        X_right, y_right = X[feature > threshold], y[feature > threshold]

        # calculate entropy after splitting
        # remember that we should use weighted entropy instead of entropy after splitting
        splitted_entropy = weighted_entropy(y_left, y_right)

        return (X_left, y_left), (X_right, y_right), splitted_entropy

    # Find the best split for the dataset and a node
    def find_best_split(self, X, y):
        # initialize
        min_entropy = np.inf
        best_feature_index = None
        best_threshold = None

        for feature_index in range(X.shape[1]):
            # all potential splits
            for threshold in self.feature_thresholds[feature_index]:
                # entropy before splitting
                # entropy after splitting
                _, _, splitted_entropy = self.split_dataset(
                    X, y, feature_index, threshold
                )

                # update if better
                if splitted_entropy < min_entropy:
                    min_entropy = splitted_entropy
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def is_pure(self, y):
        # all data belongs to the same class
        return len(np.unique(y)) == 1

    def generate(self, X, y, current_depth=0):
        # to form a leaf node, the following conditions should be met:
        # 1. the number of samples in this node is bigger than self.min_samples (must)
        # 2. reach the maximum depth
        # 3. all data in this node belongs to the same class (pure)
        # 4. no left or right subtree
        if X.shape[0] >= self.min_samples and (
            current_depth == self.max_depth or self.is_pure(y)
        ):
            return TreeNode(
                is_leaf=True,
                predicted_class=np.bincount(y).argmax(),
            )

        # find best split
        best_feature_index, best_threshold = self.find_best_split(X, y)
        (X_left, y_left), (X_right, y_right), _ = self.split_dataset(
            X, y, best_feature_index, best_threshold
        )

        # if cannot be splitted, create a leaf node
        if X_left.shape[0] == 0 or X_right.shape[0] == 0:
            return TreeNode(
                is_leaf=True,
                predicted_class=np.bincount(y).argmax(),
            )

        root = TreeNode(
            feature_index=best_feature_index,
            threshold=best_threshold,
            num_samples=len(y),
            gini=gini_index(y),
        )
        # generate left subtree
        root.left = self.generate(X_left, y_left, current_depth + 1)
        # generate right subtree
        root.right = self.generate(X_right, y_right, current_depth + 1)

        return root

    def generate_tree(self, X, y, current_depth=0):
        print("Generating tree...")
        self.root = self.generate(X, y, current_depth)
        print("Done!")

    def traverse(self, root):
        if root is None:
            return

        print(f"split on: {root.feature_index}")
        print(f"threshold: {root.threshold}")
        print(f"is leaf: {root.is_leaf}")
        print(f"predicted class: {root.predicted_class}\n")
        self.traverse(root.left)
        self.traverse(root.right)

    def compute_feature_importance(self) -> t.Sequence[float]:
        # importance = sample of node j / the number of samples * gini index of node j
        # feature importance of feature i = sum(importance of all nodes splitted on feature i)
        feature_importance = [0] * len(self.feature_thresholds)
        current_impurity = self.root.gini

        # use level order traversal
        queue = [self.root]
        while len(queue) > 0:
            node = queue.pop(0)
            # leaf node doesn't need to be considered
            if node.is_leaf:
                continue

            weight = node.num_samples / self.num_samples
            importance = weight * (current_impurity - node.gini)
            feature_importance[node.feature_index] += importance

            if node.left is not None:
                queue.append(node.left)
            if node.right is not None:
                queue.append(node.right)

        return np.array(feature_importance)


def entropy(y):
    # get the count of each unique value
    _, counts = np.unique(y, return_counts=True)
    # calculate the probability of each unique value
    probs = counts / len(y)
    return np.sum(-probs * np.log2(probs))


def weighted_entropy(y_left, y_right):
    # weight = len(y') / len(y)
    N = len(y_left) + len(y_right)
    return len(y_left) / N * entropy(y_left) + len(y_right) / N * entropy(y_right)


def gini_index(y):
    # gini index = 1 - sum(p^2)
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.sum(probs**2)
