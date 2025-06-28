import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import optuna
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from properscoring import crps_gaussian, crps_ensemble

from src.loader import (
    load_dataset_offline, clean_data,
    standardize_data, preprocess_data
)
from src.extrapolation_methods import (
    random_split, mahalanobis_split, umap_split,
    kmeans_split, gower_split, kmedoids_split, spatial_depth_handled
)
from src.evaluation_metrics import (
    evaluate_crps, evaluate_log_loss
)
from src.models.Advanced_models import (
    DistributionalRandomForestRegressor,
    LightGBMLSSRegressor,
    GAMRegressor, GAMClassifier
)
from properscoring import crps_gaussian

# --- experiment constants ---
SEED      = 10
N_TRIALS  = 5           
VAL_RATIO = 0.2        
QUANTILE_SAMPLES = 100

SUITE_CONFIG = {
    "regression_numerical":            {"suite_id":336, "task_type":"regression",     "data_type":"numerical"},
    "classification_numerical":        {"suite_id":337, "task_type":"classification", "data_type":"numerical"},
    "regression_numerical_categorical":{"suite_id":335, "task_type":"regression",     "data_type":"numerical_categorical"},
    "classification_numerical_categorical":{
        "suite_id":334, "task_type":"classification","data_type":"numerical_categorical"
    }
}
EXTRAPOLATION_METHODS = {
    "numerical":            [random_split, mahalanobis_split, kmeans_split, umap_split, spatial_depth_handled],
    "numerical_categorical":[random_split, gower_split, kmedoids_split, umap_split]
}


def find_config(suite_id):
    for cfg in SUITE_CONFIG.values():
        if cfg["suite_id"] == suite_id:
            return cfg
    raise ValueError(f"No suite config for suite_id={suite_id}")


def split_dataset(split_fn, X, y):
    """Handle random_split (6 outputs) and other 2-output splitters uniformly."""
    out = split_fn(X, y) if split_fn is random_split else split_fn(X)
    if isinstance(out, tuple) and len(out) == 6:
        X_tr, _, y_tr, _, X_te, y_te = out
    else:
        train_idx, test_idx = out
        X_tr, X_te = X.loc[train_idx], X.loc[test_idx]
        y_tr, y_te = y.loc[train_idx], y.loc[test_idx]
    return X_tr, y_tr, X_te, y_te


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite_id",     type=int,   required=True)
    parser.add_argument("--task_id",      type=int,   required=True)
    parser.add_argument("--seed",         type=int,   default=SEED)
    parser.add_argument("--result_folder",type=str,   required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg     = find_config(args.suite_id)
    methods = EXTRAPOLATION_METHODS[cfg["data_type"]]
    is_reg  = (cfg["task_type"] == "regression")

    X_full, y_full, cat_ind, attr_names = load_dataset_offline(
        args.suite_id, args.task_id
    )

    MAX_SAMPLES = 10000
    if len(X_full) > MAX_SAMPLES:
        X_full, _, y_full, _= train_test_split(
            X_full, y_full, 
            train_size=MAX_SAMPLES,
            stratify=y_full, 
            random_state=args.seed
        )

    X, X_clean, y = clean_data(
        X_full, y_full, cat_ind, attr_names,
        task_type=cfg["task_type"]
    )
    if args.task_id in (361082, 361088, 361099) and is_reg:
        y = np.log(y)

    out_dir = os.path.join(args.result_folder, f"seed_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{args.suite_id}_{args.task_id}_advanced.csv")

    records = []
    quantiles=list(np.random.uniform(0,1,QUANTILE_SAMPLES))

    for split_fn in methods:
        name_split = split_fn.__name__
        X_tr_clean, y_tr, X_te_clean, y_te = split_dataset(split_fn, X_clean, y)

        train_idx = X_tr_clean.index
        test_idx  = X_te_clean.index


        X_train_clean_unfilt, _, X_val_clean_unfilt, _ = split_dataset(split_fn, X_tr_clean, y_tr)
        train_final_idx = X_train_clean_unfilt.index
        val_idx         = X_val_clean_unfilt.index

 
        X_loop = X.copy()

        dummy_cols = X_loop.select_dtypes(include=['bool','category','object','string']).columns
        for col in dummy_cols:
            if len(X_loop[col].unique()) != len(X_loop.loc[train_final_idx, col].unique()):
                X_loop = X_loop.drop(col, axis=1)

        non_dummy_cols = X_loop.select_dtypes(
            exclude=['bool','category','object','string']
        ).columns.tolist()

        X_loop = pd.get_dummies(X_loop, drop_first=True).astype('float32')


        X_tr_p = X_loop.loc[train_idx]
        X_te_p = X_loop.loc[test_idx]
    

        X_train = X_tr_p.loc[train_final_idx]
        X_val_    = X_tr_p.loc[val_idx]
        
        y_train_ = y_tr.loc[train_final_idx].to_numpy().ravel()
        y_val    = y_tr.loc[val_idx].      to_numpy().ravel()

        X_train_, X_val = standardize_data(X_train, X_val_,non_dummy_cols)

        X_tr_p, X_te_p = standardize_data(X_tr_p, X_te_p, non_dummy_cols)
        y_tr_arr       = y_tr.to_numpy().ravel()
        y_te_arr       = y_te.to_numpy().ravel()


        def obj_crps_drf(trial):
            params = {
                'num_trees':      trial.suggest_int('num_trees', 100, 500),
                'mtry':           trial.suggest_int('mtry', 1, X_train_.shape[1]),
                'min_node_size':  trial.suggest_int('min_node_size', 10, 100),
                'seed':           args.seed
            }
            model = DistributionalRandomForestRegressor(**params)
            model.fit(X_train_, y_train_)
            y_q = model.predict_quantiles(X_val, quantiles=quantiles)
            crps_vals = [crps_ensemble(y_val[i], y_q.quantile[i].reshape(-1))
                         for i in range(len(y_val))]
            return float(np.mean(crps_vals))

        def obj_crps_lss(trial):
            params = {
                'learning_rate':    trial.suggest_float('learning_rate', 1e-4, 0.5, log=True),
                'n_estimators':     trial.suggest_int('n_estimators', 100, 500),
                'reg_lambda':       trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'max_depth':        trial.suggest_int('max_depth', 1, 30),
                'min_child_samples':trial.suggest_int('min_child_samples', 10, 100)
            }
            model = LightGBMLSSRegressor(**params)
            model.fit(X_train_.to_numpy(), y_train_)
            pr = model.predict_parameters(X_val)
            mu, sig = pr['loc'], pr['scale']
            crps_vals = crps_gaussian(y_val, mu, sig)
            return float(np.mean(crps_vals))
        
        def obj_crps_gam(trial):
            params = {
                'n_splines':     trial.suggest_int('n_splines', 10, 100),
                'spline_order':  trial.suggest_int('spline_order', 1, 5),
                'lam':           trial.suggest_float('lam',  1e-3, 1e3, log=True)
            }
            model = GAMRegressor(**params)
            model.fit(X_train_, y_train_)
            pr = model.predict_parameters(X_val)
            mu, sig = pr['loc'], pr['scale']
            return float(np.mean(crps_gaussian(y_val, mu, sig)))


        def obj_rmse_gam(trial):
            params = {
                'n_splines':     trial.suggest_int('n_splines', 10, 100),
                'spline_order':  trial.suggest_int('spline_order', 1, 5),
                'lam':           trial.suggest_float('lam',  1e-3, 1e3, log=True)
            }
            model = GAMRegressor(**params)
            model.fit(X_train_, y_train_)
            y_pred = model.predict(X_val)
            return float(np.sqrt(np.mean((y_val - y_pred)**2)))


        def obj_logloss_gam(trial):
            params = {
                'n_splines':     trial.suggest_int('n_splines', 10, 100),
                'spline_order':  trial.suggest_int('spline_order', 1, 5),
                'lam':           trial.suggest_float('lam',  1e-3, 1e3, log=True)
            }
            model = GAMClassifier(**params)
            model.fit(X_train_, y_train_)
            probs = model.predict_proba(X_val)
            return float(evaluate_log_loss(y_val, probs))


        def obj_accuracy_gam(trial):
            params = {
                'n_splines':     trial.suggest_int('n_splines', 10, 100),
                'spline_order':  trial.suggest_int('spline_order', 1, 5),
                'lam':           trial.suggest_float('lam',  1e-3, 1e3, log=True)
            }
            model = GAMClassifier(**params)
            model.fit(X_train_, y_train_)
            preds = model.predict(X_val)
            return float(accuracy_score(y_val, preds))

        model_objectives_to_run = []

        if is_reg:
            model_objectives_to_run.extend([
                ('DRF_CRPS', obj_crps_drf, 'minimize', 'CRPS', DistributionalRandomForestRegressor),
                ('LGBMLSS_CRPS', obj_crps_lss, 'minimize', 'CRPS', LightGBMLSSRegressor),
                ('GAMReg_CRPS', obj_crps_gam, 'minimize', 'CRPS', GAMRegressor),
                ('GAMReg_RMSE', obj_rmse_gam, 'minimize', 'RMSE', GAMRegressor),
            ])
        else: #
            model_objectives_to_run.extend([
                ('GAMC_LogLoss', obj_logloss_gam, 'minimize', 'LogLoss', GAMClassifier),
                ('GAMC_Accuracy', obj_accuracy_gam, 'maximize', 'Accuracy', GAMClassifier),
            ])
        studies = {}
        for name, fn, direction, metric, ModelClass in model_objectives_to_run:
            study = optuna.create_study(
                sampler=optuna.samplers.TPESampler(seed=args.seed),
                direction=direction
            )
            study.optimize(fn, n_trials=N_TRIALS)
            studies[name] = (study, metric, ModelClass)

        for name, (study, metric, ModelClass) in studies.items():
            best_params = study.best_params
            model = ModelClass(**best_params)
            model.fit(X_tr_p if ModelClass is GAMRegressor else 
                    (X_tr_p.to_numpy() if ModelClass is LightGBMLSSRegressor else X_tr_p), 
                    y_tr if ModelClass in (GAMRegressor, DistributionalRandomForestRegressor) else y_tr_arr)

            if metric == 'CRPS':
                if isinstance(model, DistributionalRandomForestRegressor):
                    y_q_test = model.predict_quantiles(X_te_p, quantiles=quantiles) 
                    forecasts_test = y_q_test.quantile
                    
                    forecasts_for_crps = forecasts_test
                    if forecasts_test.ndim == 3 and forecasts_test.shape[1] == 1:
                        forecasts_for_crps = forecasts_test.squeeze(axis=1)
                    
                    val = float(np.mean(crps_ensemble(y_te_arr, forecasts_for_crps)))
                else: 
                    pred_params_test = model.predict_parameters(X_te_p)
                    val = float(np.mean(crps_gaussian(y_te_arr, pred_params_test['loc'], pred_params_test['scale'])))
               
            elif metric == 'RMSE':
                preds = model.predict(X_te_p if ModelClass is GAMRegressor else X_te_p.to_numpy())
                val = float(np.sqrt(np.mean((y_te_arr - preds)**2)))
            elif metric == 'LogLoss':
                probs = model.predict_proba(X_te_p)
                val = float(evaluate_log_loss(y_te_arr, probs))
            elif metric == 'Accuracy':
                preds = model.predict(X_te_p)
                val = float(accuracy_score(y_te_arr, preds))
            else:
                raise ValueError(metric)

            records.append({
                'suite_id':     args.suite_id,
                'task_id':      args.task_id,
                'split_method': name_split,
                'model':        name,
                'metric':       metric,
                'value':        val
            })


    # save results
    pd.DataFrame.from_records(records).to_csv(out_file, index=False)
    print(f"Saved advanced-model results to {out_file}")


if __name__ == '__main__':
    main()
