#!/usr/bin/env python3
"""
TabPFN Experiment script mirroring baseline_experiment.py structure,
with robust missing-value handling, bounded scaling, and silent categorical encoding.
"""
import argparse
import os
import random
import warnings

import numpy as np
import pandas as pd
import torch

# missing-value handling and bounded scaling
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# --- loader & utils ---
from src.loader import load_dataset_offline, clean_data
from src.extrapolation_methods import (
    random_split,
    mahalanobis_split,
    umap_split,
    kmeans_split,
    gower_split,
    kmedoids_split,
    spatial_depth_split
) 
from src.evaluation_metrics import (
    evaluate_rmse,
    evaluate_crps,
    evaluate_accuracy,
    evaluate_log_loss
)
from Project.src.models.TabPFN_model_old import (
    TabPFNClassifierWrapper,
    TabPFNRegressorWrapper
)


# --- experiment constants ---
SEED = 10

# --- suite configuration ---
SUITE_CONFIG = {
    "regression_numerical": {"suite_id": 336, "task_type": "regression", "data_type": "numerical"},
    "classification_numerical": {"suite_id": 337, "task_type": "classification", "data_type": "numerical"},
    "regression_numerical_categorical": {"suite_id": 335, "task_type": "regression", "data_type": "numerical_categorical"},
    "classification_numerical_categorical": {"suite_id": 334, "task_type": "classification", "data_type": "numerical_categorical"}
}

EXTRAPOLATION_METHODS = {
    "numerical": [random_split, mahalanobis_split, kmeans_split, umap_split, spatial_depth_split],
    "numerical_categorical": [random_split, gower_split, kmedoids_split, umap_split]
}

def find_config(suite_id: int) -> dict:
    for cfg in SUITE_CONFIG.values():
        if cfg["suite_id"] == suite_id:
            return cfg
    raise ValueError(f"No suite config for suite_id={suite_id}")

def split_dataset(split_fn, X: pd.DataFrame, y: pd.Series):
    """Handle splitters returning 2 or 6 outputs."""
    out = split_fn(X, y) if split_fn is random_split else split_fn(X)
    if isinstance(out, tuple) and len(out) == 6:
        X_train, _, y_train, _, X_test, y_test = out
    else:
        train_idx, test_idx = out
        X_train = X.loc[train_idx]
        X_test  = X.loc[test_idx]
        y_train = y.loc[train_idx]
        y_test  = y.loc[test_idx]
    return X_train, y_train, X_test, y_test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite_id", type=int, required=True)
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--result_folder", type=str, required=True)
    args = parser.parse_args()

    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # silence sklearn encoder warnings for unseen categories
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="sklearn.preprocessing._encoders"
    )

    # output path
    seed_folder = f"seed_{args.seed}"
    out_dir = os.path.join(args.result_folder, seed_folder)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{args.suite_id}_{args.task_id}_tabpfn.csv")

    # config & splits
    cfg = find_config(args.suite_id)
    methods = EXTRAPOLATION_METHODS[cfg["data_type"]]
    is_regression = cfg["task_type"] == "regression"

    # load & clean
    X_full, y_full, cat_ind, attr_names = load_dataset_offline(
        args.suite_id, args.task_id
    )
    X, X_clean, y_clean = clean_data(
        X_full, y_full, cat_ind, attr_names,
        task_type=cfg["task_type"]
    )
    if args.task_id in (361082, 361088, 361099):
        y_clean = np.log(y_clean)

    records = []
    for split_fn in methods:
        name_split = split_fn.__name__
        X_train, y_train, X_test, y_test = split_dataset(
            split_fn, X_clean, y_clean
        )

        # flatten y
        y_train_arr = np.asarray(y_train).ravel()
        y_test_arr  = np.asarray(y_test).ravel()

        # --- bounded preprocessing pipeline ---
        # identify columns
        num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = (
            [c for c in X_train.columns if c not in num_cols]
            if cfg["data_type"] == "numerical_categorical" else []
        )

        # numeric: median impute + min-max scale to [0,1]
        num_imp = SimpleImputer(strategy="median")
        num_scl = MinMaxScaler(feature_range=(0,1))
        X_tr_num = num_scl.fit_transform(
            num_imp.fit_transform(X_train[num_cols])
        )
        X_te_num = num_scl.transform(
            num_imp.transform(X_test[num_cols])
        )

        # categorical: constant impute + one-hot (ignore unseen)
        if cat_cols:
            cat_imp = SimpleImputer(
                strategy="constant",
                fill_value="__missing__"
            )
            cat_enc = OneHotEncoder(
                handle_unknown="ignore",
                sparse=False
            )
            X_tr_cat = cat_enc.fit_transform(
                cat_imp.fit_transform(
                    X_train[cat_cols].astype(str)
                )
            )
            X_te_cat = cat_enc.transform(
                cat_imp.transform(
                    X_test[cat_cols].astype(str)
                )
            )
            X_tr_scaled = np.hstack([X_tr_num, X_tr_cat])
            X_te_scaled = np.hstack([X_te_num, X_te_cat])
        else:
            X_tr_scaled = X_tr_num
            X_te_scaled = X_te_num

        # clamp any remaining inf/nan to [0,1]
        X_tr_scaled = np.nan_to_num(
            X_tr_scaled,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )
        X_te_scaled = np.nan_to_num(
            X_te_scaled,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )

        # train & evaluate
        if is_regression:
            model = TabPFNRegressorWrapper(random_state=args.seed)
            model.fit(X_tr_scaled, y_train_arr)
            y_pred = model.predict(X_te_scaled)
            rmse = evaluate_rmse(y_test_arr, y_pred)
            residuals = y_train_arr - model.predict(X_tr_scaled)
            sigma = np.std(residuals)
            crps = evaluate_crps(
                y_test_arr,
                y_pred,
                np.full_like(y_pred, sigma)
            )
            records += [
                {"suite_id": args.suite_id, "task_id": args.task_id,
                 "split_method": name_split, "model": "TabPFNRegressor",
                 "metric": m, "value": v}
                for m, v in [("RMSE", rmse), ("CRPS", crps)]
            ]
        else:
            model = TabPFNClassifierWrapper(random_state=args.seed)
            model.fit(X_tr_scaled, y_train_arr)
            probs = model.predict_proba(X_te_scaled)
            preds = (
                (probs[:,1] >= 0.5).astype(int)
                if probs.shape[1] == 2 else
                np.argmax(probs, axis=1)
            )
            acc = evaluate_accuracy(y_test_arr, preds)
            ll  = evaluate_log_loss(y_test_arr, probs)
            records += [
                {"suite_id": args.suite_id, "task_id": args.task_id,
                 "split_method": name_split, "model": "TabPFNClassifier",
                 "metric": m, "value": v}
                for m, v in [("Accuracy", acc), ("LogLoss", ll)]
            ]

    # save results
    pd.DataFrame(records).to_csv(out_file, index=False)
    print(f"Saved TabPFN results to {out_file}")

if __name__ == "__main__":
    main()
