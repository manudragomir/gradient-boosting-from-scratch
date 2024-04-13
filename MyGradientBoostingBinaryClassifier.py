import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from utils import EPS

class MyGradientBoostingBinaryClassifier:
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

        # STEP 1 - PREDICTOR 0
        positives = y.sum()
        totals = len(y)
        p = positives / totals
        self.__first_raw_pred = p
        self.__last_raw_preds = np.array([self.__from_pred_to_log_odds(p) for _ in y])

        for epoch in range(self.n_estimators):
            # STEP 2 - COMPUTE RESIDUALS
            probabilities = np.array([self.__from_log_odds_to_pred(log_odds) for log_odds in self.__last_raw_preds])
            residuals = y - probabilities

            # STEP 3 - CREATE TREE
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            curr_gammas = self.__compute_gammas(tree, X, y, residuals, probabilities)
            self.gammas.append(curr_gammas)
            self.estimators.append(tree)

            # STEP 4 - UPDATE LAST PREDS
            curr_raw_preds = self.__apply_gammas(curr_gammas, tree.apply(X))
            self.__last_raw_preds = self.__last_raw_preds + self.lr * curr_raw_preds

    def __apply_gammas(self, gammas, leaf_indices):
        return np.array([gammas[leaf_idx] for leaf_idx in leaf_indices])

    def __compute_gammas(self, tree: DecisionTreeRegressor, X, y, residuals, probabilities):
        leafs_indices = tree.apply(X)

        sum_residuals_per_leaf = {}
        sum_mul_probs_per_leaf = {}
        for leaf_idx, sample_idx in zip(leafs_indices, range(len(y))):
            sum_residuals_per_leaf[leaf_idx] = sum_residuals_per_leaf.get(leaf_idx, 0) + residuals[sample_idx]
            sum_mul_probs_per_leaf[leaf_idx] = sum_mul_probs_per_leaf.get(leaf_idx, 0) + probabilities[sample_idx] * (1 - probabilities[sample_idx])

        gammas = {}
        unique_leafs_indices = np.unique(leafs_indices)
        for leaf_idx in unique_leafs_indices:
            gammas[leaf_idx] = sum_residuals_per_leaf[leaf_idx] / sum_mul_probs_per_leaf[leaf_idx]
        return gammas

    def __from_pred_to_log_odds(self, p):
        return np.log((p + EPS) / (1 - p + EPS))
    
    def __from_log_odds_to_pred(self, logo):
        return np.exp(logo) / (1 + np.exp(logo))

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return np.array([self.__predict_sample(x) for x in X])

    def __predict_sample(self, x):
        raw_pred = self.__first_raw_pred
        for tree, gammas in zip(self.estimators, self.gammas):
            raw_pred = raw_pred + self.lr * gammas[tree.apply([x])[0]]
        return self.__from_log_odds_to_pred(raw_pred) > self.threshold
