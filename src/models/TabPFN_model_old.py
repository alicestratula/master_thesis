import numpy as np
import torch
from tabpfn import TabPFNClassifier, TabPFNRegressor

class TabPFNClassifierWrapper:
    def __init__(self, random_state=0, ignore_pretraining_limits=True, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = TabPFNClassifier(
            random_state=random_state,
            ignore_pretraining_limits=ignore_pretraining_limits,
            device=device
        )

    def fit(self, X, y):
        # X: numpy array, y: 1d array
        self.model.fit(X, y)
        return self

    def predict(self, X, batch_size: int = 64):
        preds = []
        n = X.shape[0]
        for i in range(0, n, batch_size):
            chunk = X[i : i + batch_size]
            preds_chunk = self.model.predict(chunk)
            preds.append(preds_chunk)
        return np.concatenate(preds, axis=0)


    def predict_proba(self, X, batch_size: int = 64):
        probs = []
        n = X.shape[0]
        for i in range(0, n, batch_size):
            chunk = X[i : i + batch_size]
            probs_chunk = self.model.predict_proba(chunk)
            probs.append(probs_chunk)
        return np.vstack(probs)


class TabPFNRegressorWrapper:
    def __init__(self, random_state=0, ignore_pretraining_limits=True, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = TabPFNRegressor(
            random_state=random_state,
            ignore_pretraining_limits=ignore_pretraining_limits,
            device=device
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X, batch_size: int = 64):
        preds = []
        n = X.shape[0]
        for i in range(0, n, batch_size):
            chunk = X[i : i + batch_size]
            preds_chunk = self.model.predict(chunk)
            preds.append(preds_chunk)
        return np.concatenate(preds, axis=0)
