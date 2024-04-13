import numpy as np
import pandas as pd
from utils import CLASSIFICATION, REGRESSION


class Split:
    def __init__(self, feature_idx, threshold) -> None:
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.measure = None
        self.true_indices = None
        self.false_indices = None
    
    def match_sample(self, x):
        return x[self.feature_idx] <= self.threshold
    
    def match_batch(self, X):
        return X[X[:, self.feature_idx] <= self.threshold]
    
    def get_indices_after_split(self, X):
        true_indices = np.where(X[:, self.feature_idx] <= self.threshold)[0]
        false_indices = np.where(X[:, self.feature_idx] > self.threshold)[0]
        return true_indices, false_indices


class Leaf:
    def __init__(self, X, y, problem=REGRESSION):
        self.X = X
        self.y = y

        if problem == REGRESSION:
            self.prediction = np.mean(y)
        else:
            self.prediction = np.argmax(np.bincount(y))

    def predict(self):
        return self.prediction
    
    def predict_proba(self):
        # TODO
        return 0

class DecisionNode:
    def __init__(self, true_branch, false_branch, split: Split):
        self.split = split
        self.true_branch = true_branch
        self.false_branch = false_branch

class MyDecisionTree:
    def __init__(self, max_depth, problem=REGRESSION, max_thresholds=None) -> None:
        self.__max_depth = max_depth
        self.__problem = problem
        self.__max_thresholds = max_thresholds
        self.__root = None

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.to_numpy()
        self.__root = self.__build_tree(X, y, level=0)

    def predict_sample(self, x):
        return self.__tree_predict(x, self.__root)
    
    def predict_proba_sample(self, x):
        return self.__tree_predict(x, self.__root, proba=True)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        predictions = []
        for x in X:
            predictions.append(self.__tree_predict(x, self.__root))
        return np.array(predictions)

    def __tree_predict(self, x, node, proba=False):
        if type(node) == Leaf:
            if proba:
                return node.predict_proba()
            return node.predict()
        if node.split.match_sample(x):
            return self.__tree_predict(x, node.true_branch)
        return self.__tree_predict(x, node.false_branch)
            
    def __build_tree(self, X, y, level):
        if level == self.__max_depth:
            return Leaf(X, y, problem=self.__problem)
            
        split = self.__find_best_split(X, y)

        if self.__is_perfect_fit(y[split.true_indices]):
            true_node = Leaf(X[split.true_indices], y[split.true_indices], problem=self.__problem)
        else:
            true_node = self.__build_tree(X[split.true_indices], y[split.true_indices], level + 1)

        if self.__is_perfect_fit(y[split.false_indices]):
            false_node = Leaf(X[split.false_indices], y[split.false_indices], problem=self.__problem)
        else:
            false_node = self.__build_tree(X[split.false_indices], y[split.false_indices], level + 1)

        return DecisionNode(true_node, false_node, split)


    def __find_best_split(self, X: np.array, y: np.array):
        best_split_measure = 10000000000
        best_split = None
        for feature_idx in range(X.shape[1]):
            col = X[:, feature_idx]
            thresholds = np.unique(col)

            if self.__max_thresholds:
                thresholds = np.random.choice(thresholds, size=min(len(thresholds), self.__max_thresholds),
                                            replace=False)
            for threshold in thresholds:
                curr_split = Split(feature_idx, threshold)
                true_indices, false_indices = curr_split.get_indices_after_split(X)
                measure = self.__measure_split(y[true_indices], y[false_indices])
                if len(true_indices) == 0 or len(false_indices) == 0:
                    continue
                if measure < best_split_measure:
                    best_split_measure = measure
                    best_split = curr_split
                    best_split.measure = measure
                    best_split.true_indices = true_indices
                    best_split.false_indices = false_indices
        return best_split
    
    def __measure_split(self, trues, falses):
        if self.__problem == CLASSIFICATION:
            return self.__measure_split_gini(trues, falses)
        return self.__measure_split_variance(trues, falses)
    
    def __measure_split_gini(self, trues, falses):
        true_gini = self.gini_index(trues)
        false_gini = self.gini_index(falses)
        return (len(trues) * true_gini + len(falses) * false_gini) / (len(trues) + len(falses))
    
    def __measure_split_variance(self, trues, falses):
        true_variance = self.variance(trues)
        false_variance = self.variance(falses)
        return true_variance + false_variance
    
    def variance(self, samples):
        if len(samples) == 0:
            return 0
        mean = np.mean(samples)
        return np.mean((samples - mean) ** 2)
    
    def gini_index(self, samples):
        class_counts = np.bincount(samples)
        totals = len(samples)

        gini = 1.0
        for count in class_counts:
            gini -= (count / totals) ** 2

        return gini

    def __is_perfect_fit(self, samples):
        if self.__problem == CLASSIFICATION:
            return self.gini_index(samples) == 0
        return self.variance(samples) == 0
