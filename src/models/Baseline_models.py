import numpy as np
from sklearn.linear_model import LinearRegression as _LinearRegression
from sklearn.linear_model import LogisticRegression as _LogisticRegression

__all__ = ["LinearRegressor", "LogisticRegressor", "ConstantPredictor"]

class LinearRegressor:
    """
    Wrapper for sklearn.linear_model.LinearRegression.
    """
    def __init__(self, **kwargs):
        self.model = _LinearRegression(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        raise AttributeError("LinearRegressor does not support predict_proba")

class LogisticRegressor:
    """
    Wrapper for sklearn.linear_model.LogisticRegression.
    """
    def __init__(self, **kwargs):
        self.model = _LogisticRegression(max_iter=1000,**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        # return probability of the positive class
        return self.model.predict_proba(X)[:, 1]

class ConstantPredictor:

    def __init__(self):
        self.constant_ = None
        self.class_ = None
        self.is_classification = False

    def fit(self, X, y):
        y_arr = np.asarray(y).ravel()
        self.constant_ = np.mean(y_arr)
        unique_vals = np.unique(y_arr)
        if np.issubdtype(y_arr.dtype, np.integer) and len(unique_vals) > 1:
            self.is_classification = True
            self.classes_ = np.sort(unique_vals)
            counts = np.bincount(y_arr.astype(int))
            self.class_ = np.argmax(counts)

        else:
            self.is_classification = False
            self.class_ = float(unique_vals[0])
        return self

    def predict(self, X):
        if self.constant_ is None:
            raise ValueError("ConstantPredictor has not been fitted yet.")
        try:
            n_samples = X.shape[0]
        except Exception:
            n_samples = int(X)
        if self.is_classification:
            # return majority class for all samples
            return np.full(n_samples, self.class_, dtype=int)
        else:
            # regression: return constant mean
            return np.full(n_samples, self.constant_, dtype=float)

    def predict_proba(self, X):
        if not self.is_classification:
            raise AttributeError("predict_proba is only available for classification tasks.")
        try:
            n_samples = X.shape[0]
        except Exception:
            n_samples = int(X)
        K = len(self.classes_)
        proba = np.zeros((n_samples, K), dtype=float)
        class_idx = list(self.classes_).index(self.class_)
        proba[:, class_idx] = 1.0
        return proba

