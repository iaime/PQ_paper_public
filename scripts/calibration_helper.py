import numpy as np
import random
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
random.seed(42)

class Predictor(BaseEstimator, ClassifierMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        self.X_ = X
        self.y_ = y
        self.classes_ = unique_labels(y)
        return self

    def predict_proba(self, X):
        ret = []
        for x in X:
            pos_prob = x[0]
            neg_prob = 1 - x[0]
            ret.append([neg_prob, pos_prob])
        return np.array(ret)
    
    def predict(self, X):
        return np.array([1 if x > 0.5 else 0 for x in X])