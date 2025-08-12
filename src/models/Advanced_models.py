# src/models/advanced_models.py
import warnings
from drf import drf
from matplotlib.pylab import LinAlgError
import numpy as np
import pandas as pd
import lightgbm as lgb
from pygam import LinearGAM, LogisticGAM
import gpboost as gpb
from gpboost import GPModel
from lightgbmlss.distributions.Gaussian import Gaussian
from lightgbmlss.model import LightGBMLSS
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
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
                 gp_approx="vecchia", likelihood="gaussian", trace = True, seed=10, **kw):
        if cov_function != "matern" and cov_fct_shape is not None:
            warnings.warn("Cov_fct_shape is only used with the matern kernel, ignore it.")
            cov_fct_shape = None
        self.cov_function = cov_function
        self.cov_fct_shape = cov_fct_shape
        self.gp_approx     = gp_approx
        self.likelihood    = likelihood
        self.seed          = seed
        self.kwargs        = kw
        self._model        = None
        self.trace         = trace

    def fit(self, X, y):
        intercept = np.ones(len(y))

        gp_kwargs = {
            "gp_coords": X,
            "gp_approx": self.gp_approx,
            "cov_function": self.cov_function,
            "likelihood": self.likelihood,
            "seed": self.seed,
            **self.kwargs
        }

        if self.cov_function == "matern" and self.cov_fct_shape is not None:
            gp_kwargs["cov_fct_shape"] = self.cov_fct_shape
        self._model = gpb.GPModel(**gp_kwargs)
        self._model.fit(y=y, X=intercept, params={"trace": True})
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
    def predict_parameters(self, X):

        mu, var = self.predict(X, return_var=True)
        return {"loc": mu, "scale": np.sqrt(var)}


# ——————————————
# Classification wrappers
# ——————————————

    

class GPBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, *, cov_function="matern", cov_fct_shape=None,
                 gp_approx="vecchia", likelihood="bernoulli_logit", matrix_inversion_method = None, trace = True, seed=10, **kw):
        if cov_function != "matern" and cov_fct_shape is not None:
            warnings.warn("Cov_fct_shape is only used with the matern kernel, ignore it.")
            cov_fct_shape = None
        self.cov_function = cov_function
        self.cov_fct_shape = cov_fct_shape
        self.gp_approx     = gp_approx
        self.likelihood    = likelihood
        self.matrix_inversion_method = matrix_inversion_method
        self.seed          = seed
        self.kwargs        = kw
        self._model        = None
        self.trace         = trace

    def fit(self, X, y):
        intercept = np.ones(len(y))
        if self.gp_approx == "fitc" and self.likelihood == "bernoulli_logit":
            method = "cholesky"
        else:
            method = self.matrix_inversion_method or "iterative"

        gp_kwargs = {
            "gp_coords": X,
            "gp_approx": self.gp_approx,
            "cov_function": self.cov_function,
            "likelihood": self.likelihood,
            "matrix_inversion_method": method,
            "seed": self.seed,
            **self.kwargs
        }

        if self.cov_function == "matern" and self.cov_fct_shape is not None:
            gp_kwargs["cov_fct_shape"] = self.cov_fct_shape
        
        n=len(y)
        if gp_kwargs.get("gp_approx") == "full_scale_vecchia":
            gp_kwargs["num_ind_points"] = min(self.kwargs.get("num_ind_points", n-1), n-1)
        self._model = gpb.GPModel(**gp_kwargs)
        self._model.fit(y=y, X=intercept, params={"trace": True})
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        intercept = np.ones((X.shape[0], 1))
        out = self._model.predict(
            gp_coords_pred = X,
            X_pred         = intercept,
            predict_var    = False,
            predict_response = True
        )
        mu = out["mu"]
        mu = np.nan_to_num(mu, nan=0.5)
        return np.vstack([1 - mu, mu]).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba > 0.5).astype(int)



class GPBoostMulticlassClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._model = None
        self._label_encoder = LabelEncoder()

    def fit(self, X, y):

        self._model = OneVsRestClassifier(GPBoostClassifier(**self.kwargs))
        y_encoded = self._label_encoder.fit_transform(y)
        self._model.fit(X, y_encoded)
        return self

    def predict_proba(self, X):
        return self._model.predict_proba(X)
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)



