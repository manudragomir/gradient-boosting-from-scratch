import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from MyDecisionTree import MyDecisionTree
from utils import REGRESSION


class MyGradientBoostingRegressor:
    def __init__(self, n_estimators=10, lr=1e-1, max_depth=3, use_sklearn=False):
        self.n_estimators = n_estimators
        self.last_preds = []
        self.estimators = []
        self.lr = lr
        self.max_depth = max_depth
        self.use_sklearn = use_sklearn

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.to_numpy()

        # STEP 1 - PREDICTOR 0
        mean = np.mean(y)
        self.first_pred = mean
        self.last_preds = np.array([mean for _ in y])

        for epoch in range(self.n_estimators):
            # STEP 2 - COMPUTE RESIDUALS
            residuals = y - self.last_preds

            # STEP 3 - CREATE TREE
            if self.use_sklearn:
                tree = DecisionTreeRegressor(max_depth=self.max_depth)
            else:
                tree = MyDecisionTree(max_depth=self.max_depth, problem=REGRESSION)
            tree.fit(X, residuals)
            self.estimators.append(tree)

            # STEP 4 - UPDATE LAST PREDS
            self.last_preds = self.last_preds + self.lr * tree.predict(X)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return np.array([self.predict_sample(x) for x in X])

    def predict_sample(self, x):
        pred = self.first_pred
        for tree in self.estimators:
            prediction = tree.predict([x])[0] if self.use_sklearn else tree.predict_sample(x)
            pred = pred + self.lr * prediction
        return pred
