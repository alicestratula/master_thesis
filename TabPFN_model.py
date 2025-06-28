import numpy as np
import pandas as pd
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
        # 1) make a clean numpy array with no NaNs  
        X_np = np.asarray(pd.DataFrame(X).fillna(0), dtype=np.float32)
        y_np = np.asarray(y).ravel().astype(int)
        self.model.fit(X_np, y_np)
        return self

    def predict_proba(self, X, batch_size: int = 64):
        X_df = pd.DataFrame(X).fillna(0)
        n = X_df.shape[0]
        probs = []
        for i in range(0, n, batch_size):
            chunk = X_df.iloc[i : i + batch_size].to_numpy(dtype=np.float32)
            probs_chunk = self.model.predict_proba(chunk)
            probs.append(probs_chunk)
        return np.vstack(probs)

    def predict(self, X, batch_size: int = 64):
        probs = self.predict_proba(X, batch_size=batch_size)
        # binary vs multi-class decoding
        if probs.shape[1] == 2:
            return (probs[:, 1] >= 0.5).astype(int)
        else:
            return np.argmax(probs, axis=1)


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
        X_np = np.asarray(pd.DataFrame(X).fillna(0), dtype=np.float32)
        y_np = np.asarray(y).ravel().astype(float)
        self.model.fit(X_np, y_np)
        return self

    def predict(self, X, batch_size: int = 64):
        X_df = pd.DataFrame(X).fillna(0)
        n = X_df.shape[0]
        preds = []
        for i in range(0, n, batch_size):
            chunk = X_df.iloc[i : i + batch_size].to_numpy(dtype=np.float32)
            preds_chunk = self.model.predict(chunk)
            preds.append(preds_chunk)
        return np.concatenate(preds, axis=0)
