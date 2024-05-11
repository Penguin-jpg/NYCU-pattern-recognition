import pandas as pd
from loguru import logger

from src import AdaBoostClassifier, BaggingClassifier, DecisionTree
from src.utils import plot_learners_roc, get_accuracy, plot_feature_importance


def main():
    train_df = pd.read_csv("./train.csv")
    test_df = pd.read_csv("./test.csv")

    X_train = train_df.drop(["target"], axis=1).to_numpy()  # (n_samples, n_features)
    y_train = train_df["target"].to_numpy()  # (n_samples, )

    X_test = test_df.drop(["target"], axis=1).to_numpy()
    y_test = test_df["target"].to_numpy()

    feature_names = list(train_df.drop(["target"], axis=1).columns)

    """
    Feel free to modify the following section if you need.
    Remember to print out logs with loguru.
    """
    # AdaBoost
    clf_adaboost = AdaBoostClassifier(
        input_dim=X_train.shape[1],
    )
    _ = clf_adaboost.fit(
        X_train,
        y_train,
        # num_epochs=3000,
        # learning_rate=0.001,
        num_epochs=4000,
        learning_rate=0.06,
    )
    y_pred_classes, y_pred_probs = clf_adaboost.predict_learners(X_test)
    accuracy_ = get_accuracy(y_preds=y_pred_classes, y_trues=y_test)
    logger.info(f"AdaBoost - Accuracy: {accuracy_:.4f}")
    plot_learners_roc(
        y_preds=y_pred_probs,
        y_trues=y_test,
        fpath="adaboost.png",
    )
    feature_importance = clf_adaboost.compute_feature_importance()
    plot_feature_importance(feature_names, feature_importance)

    # Bagging
    clf_bagging = BaggingClassifier(
        input_dim=X_train.shape[1],
    )
    _ = clf_bagging.fit(
        X_train,
        y_train,
        num_epochs=3000,
        learning_rate=0.001,
    )
    y_pred_classes, y_pred_probs = clf_bagging.predict_learners(X_test)
    accuracy_ = get_accuracy(y_preds=y_pred_classes, y_trues=y_test)
    logger.info(f"Bagging - Accuracy: {accuracy_:.4f}")
    plot_learners_roc(
        y_preds=y_pred_probs,
        y_trues=y_test,
        fpath="bagging.png",
    )
    feature_importance = clf_bagging.compute_feature_importance()
    plot_feature_importance(feature_names, feature_importance)

    # Decision Tree
    clf_tree = DecisionTree(
        max_depth=7,
    )
    clf_tree.fit(X_train, y_train)
    y_pred_classes = clf_tree.predict(X_test)
    accuracy_ = get_accuracy(y_preds=y_pred_classes, y_trues=y_test)
    logger.info(f"DecisionTree - Accuracy: {accuracy_:.4f}")
    plot_learners_roc(
        y_preds=y_pred_classes,
        y_trues=y_test,
        fpath="decision_tree.png",
    )
    feature_importance = clf_tree.compute_feature_importance()
    plot_feature_importance(feature_names, feature_importance)


if __name__ == "__main__":
    main()
