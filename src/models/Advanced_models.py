# src/models/advanced_models.py
import warnings
from drf import drf
from matplotlib.pylab import LinAlgError
import numpy as np
import pandas as pd
import lightgbm as lgb
from pygam import LinearGAM, LogisticGAM
import gpboost as gpb
from lightgbmlss.distributions.Gaussian import Gaussian
from lightgbmlss.model import LightGBMLSS
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder

# ——————————————
# Regression wrappers
# ——————————————

class DistributionalRandomForestRegressor:
    def __init__(self, *, num_trees=100, mtry=None, min_node_size=5, seed=0, **kwargs):
        self.num_trees     = num_trees
        self.mtry          = mtry
        self.min_node_size = min_node_size
        self.seed          = seed
        self.kwargs        = kwargs
        self._model        = None


    def fit(self, X, y): 
        drf_hyperparams = {
            'num_trees': self.num_trees,
            'mtry': self.mtry if self.mtry is not None else X.shape[1], # Default mtry
            'min_node_size': self.min_node_size,
            'seed': self.seed
        }

        drf_hyperparams.update(self.kwargs)

        self._model = drf(**drf_hyperparams) 

        self._model.fit(X, y) 
        return self

    def predict_quantiles(self, X, quantiles):
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() before predict_quantiles().")
        return self._model.predict(newdata=X, functional="quantile", quantiles=quantiles)

    def predict(self, X):
        # median
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() before predict().")
        q = self.predict_quantiles(X, quantiles=[0.5])
        return q.quantile[:, 0]



class LightGBMLSSRegressor:
    def __init__(self, *, distribution=None, **opts):
        self.distribution = distribution or Gaussian(
            stabilization="None", response_fn="exp", loss_fn="crps"
        )
        self.opts = opts
        self._model = None

    def fit(self, X, y):
        dtrain = lgb.Dataset(X, label=y)
        num_round = self.opts.pop("n_estimators",
                     self.opts.pop("num_boost_round", 100))
        self.opts.setdefault("feature_pre_filter", False)
        self._model = LightGBMLSS(self.distribution)
        self._model.train(self.opts, dtrain, num_boost_round=num_round)
        return self

    def predict_parameters(self, X):
        return self._model.predict(X, pred_type="parameters")

    def predict_mean(self, X):
        return self.predict_parameters(X)["loc"]

    def predict_std(self, X):
        return self.predict_parameters(X)["scale"]
    



class GAMRegressor:
    def __init__(self, n_splines=25, spline_order=3, lam=0.6, **kwargs):

        self.n_splines = n_splines
        self.spline_order = spline_order
        self.lam = lam
        self.kwargs = kwargs
        self._model = None
        self.std_dev_error_ = None  

    def fit(self, X, y):
        X_np = X.to_numpy() if isinstance(X, (pd.DataFrame, pd.Series)) else np.asarray(X)
        y_np = y.to_numpy().ravel() if isinstance(y, (pd.DataFrame, pd.Series)) else np.asarray(y).ravel()

        self._model = LinearGAM(
            n_splines=self.n_splines,
            spline_order=self.spline_order,
            lam=self.lam,
            **self.kwargs
        )
        try:
            self._model.fit(X_np, y_np)
        except LinAlgError as e:
            warnings.warn(f"SVD did not converge with lam= {self.lam}. ")
            self.lam *= 10 
            self._model = LinearGAM(
                n_splines=self.n_splines,
                spline_order=self.spline_order,
                lam=self.lam,
                **self.kwargs
            )
            self._model.fit(X_np, y_np)

        train_predictions = self._model.predict(X_np)
        residuals = y_np - train_predictions
        self.std_dev_error_ = np.std(residuals)
        
        if self.std_dev_error_ == 0 or np.isnan(self.std_dev_error_):
            self.std_dev_error_ = 1e-6  
        return self

    def predict(self, X):
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        X_np = X.to_numpy() if isinstance(X, (pd.DataFrame, pd.Series)) else np.asarray(X)
        return self._model.predict(X_np)

    def predict_parameters(self, X):
        if self._model is None or self.std_dev_error_ is None:
            raise RuntimeError("Model has not been fitted yet or std_dev_error_ not computed. Call fit() first.")
        
        mu = self.predict(X)
        sigma = np.full_like(mu, self.std_dev_error_)
    
        return {'loc': mu, 'scale': sigma}


class GPBoostRegressor:
    def __init__(self, *, cov_function="matern", cov_fct_shape=None,
                 gp_approx="vecchia", likelihood="gaussian", seed=0, **kw):
        self.cov_function = cov_function
        self.cov_fct_shape = cov_fct_shape
        self.gp_approx     = gp_approx
        self.likelihood    = likelihood
        self.seed          = seed
        self.kwargs        = kw
        self._model        = None

    def fit(self, X, y):
        intercept = np.ones(len(y))
        self._model = gpb.GPModel(
            gp_coords=X,
            cov_function=self.cov_function,
            cov_fct_shape=self.cov_fct_shape,
            gp_approx=self.gp_approx,
            likelihood=self.likelihood,
            seed=self.seed,
            **self.kwargs
        )
        self._model.fit(y=y, X=intercept, params={"trace": False})
        return self

    def predict(self, X, return_var=False):
        intercept = np.ones(X.shape[0])
        out = self._model.predict(
            gp_coords_pred=X,
            X_pred=intercept,
            predict_var=return_var,
            predict_response=True
        )
        mu = out["mu"]
        if return_var:
            return mu, out["var"]
        return mu


# ——————————————
# Classification wrappers
# ——————————————



class GAMClassifier:
    def __init__(self, n_splines=25, spline_order=3, lam=0.6, **kwargs):
        self.n_splines = n_splines
        self.spline_order = spline_order
        self.lam = lam
        self.kwargs = kwargs
        self._model = None
        self.classes_ = None

    def fit(self, X, y):
        X_np = X.to_numpy() if isinstance(X, (pd.DataFrame, pd.Series)) else np.asarray(X)
        y_np = y.to_numpy().ravel()   if isinstance(y, (pd.DataFrame, pd.Series)) else np.asarray(y).ravel()

        le = LabelEncoder().fit(y_np)
        y_enc  = le.transform(y_np)
        self.classes_ = le.classes_

        base_gam = LogisticGAM(
            n_splines=self.n_splines,
            spline_order=self.spline_order,
            lam=self.lam,
            **self.kwargs
        )
        
        if self.classes_.size > 2:
            self._model = OneVsRestClassifier(base_gam)
            fit_y = y_np
        else:
            self._model = base_gam
            fit_y = y_enc

        self._model.fit(X_np, fit_y)
        return self

    def predict_proba(self, X):
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        X_np = X.to_numpy() if isinstance(X, (pd.DataFrame, pd.Series)) else np.asarray(X)

        proba = self._model.predict_proba(X_np)
        if isinstance(self._model, OneVsRestClassifier):
            cols = []
            for est in self._model.estimators_:
                p1 = est.predict_proba(X_np)        
                p1 = np.nan_to_num(p1, nan=0.5)
                cols.append(p1)
            return np.vstack(cols).T

        p1 = self._model.predict_proba(X_np)         
        p1 = np.nan_to_num(p1, nan=0.5)
        return np.vstack([1 - p1, p1]).T


    def predict(self, X):
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]


class GPBoostClassifier:
    """
    Wrapper around gpboost.GPModel for binary classification
    using Bernoulli‐logit likelihood.
    """
    def __init__(self, *, cov_function="matern", cov_fct_shape=None,
                 gp_approx="vecchia", seed=10, **kwargs):
        self.cov_function = cov_function
        self.cov_fct_shape = cov_fct_shape
        self.gp_approx     = gp_approx
        self.likelihood    = "bernoulli_logit"
        self.seed          = seed
        self.kwargs        = kwargs
        self._model        = None

    def fit(self, X, y):
        intercept = np.ones(len(y))
        self._model = gpb.GPModel(
            gp_coords=X,
            cov_function=self.cov_function,
            cov_fct_shape=self.cov_fct_shape,
            gp_approx=self.gp_approx,
            likelihood=self.likelihood,
            seed=self.seed,
            **self.kwargs
        )
        self._model.fit(y=y, X=intercept, params={"trace": False})
        return self

    def predict_proba(self, X):
        intercept = np.ones(X.shape[0])
        out = self._model.predict(
            gp_coords_pred=X,
            X_pred=intercept,
            predict_var=False,
            predict_response=True
        )
        # 'mu' is already P(y=1)
        probs = out["mu"]
        # fill any NaNs just in case
        probs = np.nan_to_num(probs, nan=0.5)
        return probs

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)



