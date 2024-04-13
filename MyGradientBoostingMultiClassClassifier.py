import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from utils import EPS


class MyGradientBoostingMultiClassClassifier:
    def __init__(self, n_estimators=10, lr=1e-2, max_depth=3):
        self.n_estimators = n_estimators
        self.estimators = []
        self.gammas = []
        self.lr = lr
        self.max_depth = max_depth
        self.threshold = 0.5

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.to_numpy()

        # STEP 0 - convert y to yohe
        rows = len(y)
        class_count = np.bincount(y)
        no_classes = len(class_count)
        yohe = self.__to_one_hot_encoded(y, no_classes)

        # STEP 1 - PREDICTOR 0
        probabilities = class_count / rows
        self.__first_raw_pred = [self.__from_pred_to_log_odds(p) for p in probabilities]
        self.__last_raw_preds = np.array([self.__first_raw_pred for _ in y])

        for epoch in range(self.n_estimators):
            # STEP 2 - COMPUTE RESIDUALS
            probabilities = np.array([self.__from_log_odds_to_pred_array(log_odds_arr) for log_odds_arr in self.__last_raw_preds])
            residuals = yohe - probabilities

            # STEP 3 - CREATE TREE
            self.gammas.append([])
            self.estimators.append([])
            for k_class in range(no_classes):
                tree = DecisionTreeRegressor(max_depth=self.max_depth)
                tree.fit(X, residuals[:, k_class])
                curr_gammas = self.__compute_gammas(tree, X, residuals[:, k_class], probabilities[:, k_class])
                self.gammas[-1].append(curr_gammas)
                self.estimators[-1].append(tree)

                # STEP 4 - UPDATE LAST PREDS
                curr_raw_preds = self.__apply_gammas(curr_gammas, tree.apply(X))
                self.__last_raw_preds[:, k_class] = self.__last_raw_preds[:, k_class] + self.lr * curr_raw_preds

    def __to_one_hot_encoded(self, y, no_classes):
        # suppose y is an array with classes from 0 to n - 1
        rows = len(y)
        yohe = np.zeros((rows, no_classes), dtype=int)
        yohe[np.arange(rows), y] = 1
        return yohe

    def __apply_gammas(self, gammas, leaf_indices):
        return np.array([gammas[leaf_idx] for leaf_idx in leaf_indices])

    def __compute_gammas(self, tree: DecisionTreeRegressor, X, residuals, probabilities):
        leafs_indices = tree.apply(X)
        no_samples = X.shape[0]

        sum_residuals_per_leaf = {}
        sum_mul_probs_per_leaf = {}
        for leaf_idx, sample_idx in zip(leafs_indices, range(no_samples)):
            sum_residuals_per_leaf[leaf_idx] = sum_residuals_per_leaf.get(leaf_idx, 0) + residuals[sample_idx]
            sum_mul_probs_per_leaf[leaf_idx] = sum_mul_probs_per_leaf.get(leaf_idx, 0) + probabilities[sample_idx] * (1 - probabilities[sample_idx])

        gammas = {}
        unique_leafs_indices = np.unique(leafs_indices)
        for leaf_idx in unique_leafs_indices:
            gammas[leaf_idx] = sum_residuals_per_leaf[leaf_idx] / sum_mul_probs_per_leaf[leaf_idx]
        return gammas
    
    def __from_log_odds_to_pred_array(self, p_array):
        return np.array([self.__from_log_odds_to_pred(p) for p in p_array])

    # from probability space to raw space
    def __from_pred_to_log_odds(self, p):
        return np.log((p + EPS) / (1 - p + EPS))
    
    # from raw space to probability space
    def __from_log_odds_to_pred(self, logo):
        return np.exp(logo) / (1 + np.exp(logo))

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return np.array([self.__predict_sample(x) for x in X])

    def __predict_sample(self, x):
        raw_pred = np.copy(self.__first_raw_pred)
        for trees, gammas in zip(self.estimators, self.gammas):
            for class_idx, (tree, gamma) in enumerate(zip(trees, gammas)):
                raw_pred[class_idx] = raw_pred[class_idx] + self.lr * gamma[tree.apply([x])[0]]
        probabilities = self.__from_log_odds_to_pred_array(raw_pred)
        return np.argmax(probabilities)
