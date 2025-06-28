import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
import optuna
from sklearn.model_selection import train_test_split
from properscoring import crps_gaussian, crps_ensemble
from src.loader import (
    load_dataset_offline,
    clean_data,
    standardize_data,
    preprocess_data,
)
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
    evaluate_accuracy,
    evaluate_log_loss,
)
from src.models.Neural_models import (
    EngressionRegressor,
    EngressionClassifier,
    MLPRegressor,
    MLPClassifier,
    ResNetRegressor,
    ResNetClassifier,
    FTTrans_Regressor,
    FTTrans_Classifier
)

# Experiment constants
SEED = 10
N_TRIALS = 100  
VAL_RATIO = 0.2  
N_SAMPLES = 100  
PATIENCE = 40  
N_EPOCHS = 1000  
BATCH_SIZE = 1024


# Suite configuration
SUITE_CONFIG = {
    "regression_numerical": {
        "suite_id": 336,
        "task_type": "regression",
        "data_type": "numerical",
    },
    "classification_numerical": {
        "suite_id": 337,
        "task_type": "classification",
        "data_type": "numerical",
    },
    "regression_numerical_categorical": {
        "suite_id": 335,
        "task_type": "regression",
        "data_type": "numerical_categorical",
    },
    "classification_numerical_categorical": {
        "suite_id": 334,
        "task_type": "classification",
        "data_type": "numerical_categorical",
    },
}
EXTRAPOLATION_METHODS = {
    "numerical": [random_split, mahalanobis_split, kmeans_split, umap_split, spatial_depth_split],
    "numerical_categorical": [random_split, gower_split, kmedoids_split, umap_split],
}

# Helpers
def find_config(suite_id):
    for cfg in SUITE_CONFIG.values():
        if cfg["suite_id"] == suite_id:
            return cfg
    raise ValueError(f"No suite config for suite_id={suite_id}")

# Handle random_split API
def split_dataset(split_fn, X, y):
    out = split_fn(X, y) if split_fn is random_split else split_fn(X)
    if isinstance(out, tuple) and len(out) == 6:
        X_tr, _, y_tr, _, X_te, y_te = out
    else:
        tr_idx, te_idx = out
        X_tr, X_te = X.loc[tr_idx], X.loc[te_idx]
        y_tr, y_te = y.loc[tr_idx], y.loc[te_idx]
    return X_tr, y_tr, X_te, y_te

# Main experiment
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite_id", type=int, required=True)
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--result_folder", type=str, required=True)
    args = parser.parse_args()

    # Repro
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = find_config(args.suite_id)
    methods = EXTRAPOLATION_METHODS[cfg["data_type"]]
    is_reg = cfg["task_type"] == "regression"

    # Load & clean
    X_full, y_full, cat_ind, attr_names = load_dataset_offline(
        args.suite_id, args.task_id
    )
    if len(X_full) > 10000:
        X_full, _, y_full, _ = train_test_split(
            X_full,
            y_full,
            train_size=10000,
            stratify=None if is_reg else y_full,
            random_state=args.seed,
        )
    X, X_clean, y_clean = clean_data(
        X_full, y_full, cat_ind, attr_names, task_type=cfg["task_type"]
    )
    if is_reg and args.task_id in (361082, 361088, 361099):
        y_clean = np.log(y_clean)

    # Prepare output
    out_dir = os.path.join(args.result_folder, f"seed_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{args.suite_id}_{args.task_id}_neurals.csv")

    records = []

    # Loop over splits
    for split_fn in methods:
        name_split = split_fn.__name__
        X_tr, y_tr, X_te, y_te = split_dataset(split_fn, X_clean, y_clean)
        y_tr_arr = y_tr.to_numpy().ravel()
        y_te_arr = y_te.to_numpy().ravel()

        # Preprocess
        if cfg["data_type"] == "numerical_categorical":
            X_tr_p, X_te_p = preprocess_data(X_tr, X_te, X_clean)
        else:
            X_tr_p, X_te_p = standardize_data(X_tr, X_te)

        # Validation carve
        skw = dict(test_size=VAL_RATIO, random_state=args.seed)
        if not is_reg:
            skw["stratify"] = y_tr_arr
        X_train_, X_val, y_train_, y_val = train_test_split(
            X_tr_p, y_tr_arr, **skw
        )

        # --- Engression tuning ---
        if is_reg:
            # RMSE
            def eng_rmse(trial):
                checkpoint_path = os.path.join(  # Unique checkpoint path
                    out_dir, f"checkpoint_engression_rmse_{trial.number}.pt"
                )
                m = EngressionRegressor(
                    learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
                    num_epochs=trial.suggest_int("num_epochs", 100, 1000),
                    num_layer=trial.suggest_int("num_layer", 2, 5),
                    hidden_dim=trial.suggest_int("hidden_dim", 100, 500),
                    resblock=trial.suggest_categorical("resblock", [True, False]),
                    batch_size=BATCH_SIZE,
                    seed=args.seed,
                )
                
                m.fit(X_train_, y_train_)
                mu = m.predict(X_val)
                return evaluate_rmse(y_val, mu)

            study_er = optuna.create_study(
                sampler=optuna.samplers.TPESampler(seed=args.seed),
                direction="minimize",
            )
            study_er.optimize(eng_rmse, n_trials=N_TRIALS)

            # CRPS
            def eng_crps(trial):
                checkpoint_path = os.path.join(  # Unique checkpoint path
                    out_dir, f"checkpoint_engression_crps_{trial.number}.pt"
                )
                m = EngressionRegressor(
                    learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
                    num_epochs=trial.suggest_int("num_epochs", 100, 1000),
                    num_layer=trial.suggest_int("num_layer", 2, 5),
                    hidden_dim=trial.suggest_int("hidden_dim", 100, 500),
                    resblock=trial.suggest_categorical("resblock", [True, False]),
                    batch_size=BATCH_SIZE,
                    seed=args.seed,
                )
                tr_loader, val_loader, _ = m.prepare_data(
                    X_train_, y_train_, X_val, y_val, None, None
                )
                m.fit(X_train_, y_train_)

                y_val_samples = m.predict_samples(X_val, sample_size=N_SAMPLES) 
                crps_values = [
                    crps_ensemble(y_val[i], y_val_samples[i])
                    for i in range(len(y_val))
                ]
                return float(np.mean(crps_values))

            study_ec = optuna.create_study(
                sampler=optuna.samplers.TPESampler(seed=args.seed),
                direction="minimize",
            )
            study_ec.optimize(eng_crps, n_trials=N_TRIALS)

            # Refit & evaluate Engression
            for name, study, metric in [
                ("Engression", study_er, "RMSE"),
                ("Engression", study_ec, "CRPS"),
            ]:
                bp = study.best_params
                checkpoint_path = os.path.join(
                    out_dir, f"checkpoint_engression_{metric.lower()}.pt"
                )  # Use a consistent name here
                m = EngressionRegressor(
                    learning_rate=bp["learning_rate"],
                    num_epochs=bp["num_epochs"],
                    num_layer=bp["num_layer"],
                    hidden_dim=bp["hidden_dim"],
                    resblock=bp["resblock"],
                    batch_size=BATCH_SIZE,
                    seed=args.seed,
                )
                tr_loader, _, _ = m.prepare_data(
                    X_train_, y_train_, None, None, None, None
                )
                m.fit(X_tr_p, y_tr_arr)
                if metric == "RMSE":
                    mu_test = m.predict(X_te_p)
                    val = evaluate_rmse(y_te_arr, mu_test)
                else:
                    y_test_samples = m.predict_samples(X_te_p, sample_size=N_SAMPLES)
                    crps_values = [
                        crps_ensemble(y_te_arr[i], y_test_samples[i])
                        for i in range(len(y_te_arr))
                    ]
                    val = float(np.mean(crps_values))
                    
                records.append(
                    {
                        "suite_id": args.suite_id,
                        "task_id": args.task_id,
                        "split_method": name_split,
                        "model": name,
                        "metric": metric,
                        "value": val,
                    }
                )
        else:
            # Engression classification
            def eng_acc(trial):
                checkpoint_path = os.path.join(  # Unique checkpoint path
                    out_dir, f"checkpoint_engression_acc_{trial.number}.pt"
                )
                m = EngressionClassifier(
                    learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
                    num_epochs=trial.suggest_int("num_epochs", 100, 1000),
                    num_layer=trial.suggest_int("num_layer", 2, 5),
                    hidden_dim=trial.suggest_int("hidden_dim", 100, 500),
                    resblock=trial.suggest_categorical("resblock", [True, False]),
                    batch_size=BATCH_SIZE,
                    seed=args.seed,
                )
                # EngressionClassifier.fit only takes train data
                m.fit(X_train_, y_train_)
                preds = m.predict(X_val)
                return evaluate_accuracy(y_val, preds)

            def eng_ll(trial):
                checkpoint_path = os.path.join(  # Unique checkpoint path
                    out_dir, f"checkpoint_engression_ll_{trial.number}.pt"
                )
                m = EngressionClassifier(
                    learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
                    num_epochs=trial.suggest_int("num_epochs", 100, 1000),
                    num_layer=trial.suggest_int("num_layer", 2, 5),
                    hidden_dim=trial.suggest_int("hidden_dim", 100, 500),
                    resblock=trial.suggest_categorical("resblock", [True, False]),
                    batch_size=BATCH_SIZE,
                    seed=args.seed,
                )
                # EngressionClassifier.fit only takes train data
                m.fit(X_train_, y_train_)
                probs = m.predict_proba(X_val)
                return evaluate_log_loss(y_val, probs)

            st_acc = optuna.create_study(
                sampler=optuna.samplers.TPESampler(seed=args.seed),
                direction="maximize",
            )
            st_acc.optimize(eng_acc, n_trials=N_TRIALS)
            st_ll = optuna.create_study(
                sampler=optuna.samplers.TPESampler(seed=args.seed),
                direction="minimize",
            )
            st_ll.optimize(eng_ll, n_trials=N_TRIALS)
            for name, study, metric in [
                ("Engression", st_acc, "Accuracy"),
                ("Engression", st_ll, "LogLoss"),
            ]:
                bp = study.best_params
                checkpoint_path = os.path.join(
                    out_dir, f"checkpoint_engression_{metric.lower()}.pt"
                )
                m = EngressionClassifier(
                    learning_rate=bp["learning_rate"],
                    num_epochs=bp["num_epochs"],
                    num_layer=bp["num_layer"],
                    hidden_dim=bp["hidden_dim"],
                    resblock=bp["resblock"],
                    batch_size=BATCH_SIZE,
                    seed=args.seed,
                )
                m.fit(X_tr_p, y_tr_arr)
                if metric == "Accuracy":
                    preds = m.predict(X_te_p)
                    val = evaluate_accuracy(y_te_arr, preds)
                else:
                    probs = m.predict_proba(X_te_p)
                    val = evaluate_log_loss(y_te_arr, probs)
                records.append(
                    {
                        "suite_id": args.suite_id,
                        "task_id": args.task_id,
                        "split_method": name_split,
                        "model": name,
                        "metric": metric,
                        "value": val,
                    }
                )

        # --- MLP tuning ---
        if is_reg:
            # RMSE study
            # In neural_experiment.py, inside the 'mlp_rmse' Optuna objective:
            def mlp_rmse(trial):
                checkpoint_path = os.path.join(  # Unique checkpoint path
                    out_dir, f"checkpoint_mlp_rmse_{trial.number}.pt"
                )
                m = MLPRegressor(
                    n_blocks=trial.suggest_int("n_blocks", 1, 5),
                    d_block=trial.suggest_int("d_block", 10, 500), # You'll adjust ranges for testing
                    dropout=trial.suggest_float("dropout", 0, 1),
                    # Ensure MLPRegressor's __init__ expects 'learning_rate' (your latest version does)
                    learning_rate=trial.suggest_float("learning_rate", 1e-4, 5e-2, log=True),
                    weight_decay=trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
                    n_epochs=N_EPOCHS, # This is the max_epochs for this HPO trial
                    patience=PATIENCE, # This will be used by EarlyStopping inside fit
                    batch_size=BATCH_SIZE,
                    seed=args.seed,
                    checkpoint_path=checkpoint_path,
                )

                actual_epochs_trained = m.fit(X_train_, y_train_, X_val, y_val)

                trial.suggest_int('n_epochs', actual_epochs_trained, actual_epochs_trained)


                mu = m.predict(X_val)
                rmse = evaluate_rmse(y_val, mu)
                return rmse


            # CRPS study
            def mlp_crps(trial):
                checkpoint_path = os.path.join(  # Unique checkpoint path
                    out_dir, f"checkpoint_mlp_crps_{trial.number}.pt"
                )
                m = MLPRegressor(
                    n_blocks=trial.suggest_int("n_blocks", 1, 5),
                    d_block=trial.suggest_int("d_block", 10, 500),
                    dropout=trial.suggest_float("dropout", 0, 1),
                    learning_rate=trial.suggest_float("learning_rate", 1e-4, 5e-2, log=True),
                    weight_decay=trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
                    n_epochs=N_EPOCHS,
                    patience=PATIENCE,
                    batch_size=BATCH_SIZE,
                    seed=args.seed,
                    checkpoint_path=checkpoint_path,  # Pass it here
                )
                tr_loader, val_loader, _ = m.prepare_data(
                    X_train_, y_train_, X_val, y_val, None, None
                )

                actual_epochs_trained = m.fit(X_train_, y_train_, X_val, y_val)
                trial.suggest_int('n_epochs', actual_epochs_trained, actual_epochs_trained)

                mu, sigma = m.predict_with_uncertainty(X_val, tr_loader)
                return float(np.mean(crps_gaussian(y_val, mu, sigma)))

            st_rmse = optuna.create_study(
                sampler=optuna.samplers.TPESampler(seed=args.seed),
                direction="minimize",
            )
            st_crps = optuna.create_study(
                sampler=optuna.samplers.TPESampler(seed=args.seed),
                direction="minimize",
            )
            st_rmse.optimize(mlp_rmse, n_trials=N_TRIALS)
            st_crps.optimize(mlp_crps, n_trials=N_TRIALS)
            # Refit & evaluate MLP
            for name, study, metric in [
                ("MLP", st_rmse, "RMSE"),
                ("MLP", st_crps, "CRPS"),
            ]:
                bp = study.best_params
                checkpoint_path = os.path.join(
                    out_dir, f"checkpoint_mlp_{metric.lower()}.pt"
                ) 
                m = MLPRegressor(
                    **bp,
                    patience=PATIENCE,
                    batch_size=BATCH_SIZE,
                    seed=args.seed,
                    checkpoint_path=checkpoint_path,  # Pass it here
                )
                tr_loader_final, _, _ = m.prepare_data(
                    X_tr_p, y_tr_arr, None, None, None, None
                )
                m.fit(X_tr_p, y_tr_arr)
                if metric == "RMSE":
                    mu_test = m.predict(X_te_p)
                    val = evaluate_rmse(y_te_arr, mu_test)
                else:
                    mu_test, sigma_test = m.predict_with_uncertainty(X_te_p, tr_loader_final)
                    val = float(np.mean(crps_gaussian(y_te_arr, mu_test, sigma_test)))
                records.append(
                    {
                        "suite_id": args.suite_id,
                        "task_id": args.task_id,
                        "split_method": name_split,
                        "model": name,
                        "metric": metric,
                        "value": val,
                    }
                )
        else:
            # MLP Classification: LogLoss & Accuracy
            def mlp_ll(trial):
                checkpoint_path = os.path.join(  # Unique checkpoint path
                    out_dir, f"checkpoint_mlp_ll_{trial.number}.pt"
                )
                m = MLPClassifier(
                    n_blocks=trial.suggest_int("n_blocks", 1, 5),
                    d_block=trial.suggest_int("d_block", 10, 500),
                    dropout=trial.suggest_float("dropout", 0, 1),
                    learning_rate=trial.suggest_float("learning_rate", 1e-4, 5e-2, log=True),
                    weight_decay=trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
                    n_epochs=N_EPOCHS,
                    patience=PATIENCE,
                    batch_size=BATCH_SIZE,
                    seed=args.seed,
                    checkpoint_path=checkpoint_path,
                )
                actual_epochs_trained = m.fit(X_train_, y_train_, X_val, y_val)
                trial.suggest_int('n_epochs', actual_epochs_trained, actual_epochs_trained)
                probs = m.predict_proba(X_val)
                return evaluate_log_loss(y_val, probs)

            def mlp_acc(trial):
                checkpoint_path = os.path.join(  # Unique checkpoint path
                    out_dir, f"checkpoint_mlp_acc_{trial.number}.pt"
                )
                m = MLPClassifier(
                    n_blocks=trial.suggest_int("n_blocks", 1, 5),
                    d_block=trial.suggest_int("d_block", 10, 500),
                    dropout=trial.suggest_float("dropout", 0, 1),
                    learning_rate=trial.suggest_float("learning_rate", 1e-4, 5e-2, log=True),
                    weight_decay=trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
                    n_epochs=N_EPOCHS,
                    patience=PATIENCE,
                    batch_size=BATCH_SIZE,
                    seed=args.seed,
                    checkpoint_path=checkpoint_path,
                )
                actual_epochs_trained = m.fit(X_train_, y_train_, X_val, y_val)
                trial.suggest_int('n_epochs', actual_epochs_trained, actual_epochs_trained)
                preds = m.predict(X_val)
                return evaluate_accuracy(y_val, preds)

            st_ll = optuna.create_study(
                sampler=optuna.samplers.TPESampler(seed=args.seed),
                direction="minimize",
            )
            st_acc = optuna.create_study(
                sampler=optuna.samplers.TPESampler(seed=args.seed),
                direction="maximize",
            )
            st_ll.optimize(mlp_ll, n_trials=N_TRIALS)
            st_acc.optimize(mlp_acc, n_trials=N_TRIALS)
            for name, study, metric in [
                ("MLP", st_ll, "LogLoss"),
                ("MLP", st_acc, "Accuracy"),
            ]:
                bp = study.best_params
                checkpoint_path = os.path.join(
                    out_dir, f"checkpoint_mlp_{metric.lower()}.pt"
                )
                m = MLPClassifier(
                    **bp,
                    patience=PATIENCE,
                    batch_size=BATCH_SIZE,
                    seed=args.seed,
                    checkpoint_path=checkpoint_path,
                )
                m.fit(X_tr_p, y_tr_arr)
                if metric == "LogLoss":
                    probs = m.predict_proba(X_te_p)
                    val = evaluate_log_loss(y_te_arr, probs)
                else:
                    preds = m.predict(X_te_p)
                    val = evaluate_accuracy(y_te_arr, preds)
                records.append(
                    {
                        "suite_id": args.suite_id,
                        "task_id": args.task_id,
                        "split_method": name_split,
                        "model": name,
                        "tuned_for": metric,
                        "metric": metric,
                        "value": val,
                    }
                )
        #ResNet
        if is_reg:
            def resnet_rmse(trial):
                checkpoint_path = os.path.join(out_dir, f"checkpoint_resnet_rmse_{trial.number}.pt")
                m = ResNetRegressor(
                    n_blocks=trial.suggest_int("n_blocks", 1, 5),
                    d_block=trial.suggest_int("d_block", 10, 500),
                    dropout1=trial.suggest_float("dropout1", 0, 1),
                    dropout2=trial.suggest_float("dropout2", 0, 1),
                    d_hidden_multiplier=trial.suggest_float("d_hidden_multiplier", 0.5, 3),
                    learning_rate=trial.suggest_float("learning_rate", 1e-4, 5e-2, log=True),
                    weight_decay=trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
                    n_epochs=N_EPOCHS,
                    patience=PATIENCE,
                    batch_size=BATCH_SIZE,
                    seed=args.seed,
                    checkpoint_path=checkpoint_path
                )
                
                epochs_run_trained = m.fit(X_train_, y_train_,X_val, y_val)
                trial.suggest_int('n_epochs', epochs_run_trained, epochs_run_trained)

                mu = m.predict(X_val)
                rmse = evaluate_rmse(y_val, mu)
                return rmse
            
            def resnet_crps(trial):
                checkpoint_path = os.path.join(  # Unique checkpoint path
                    out_dir, f"checkpoint_resnet_crps_{trial.number}.pt"
                )
                m = ResNetRegressor(
                    n_blocks=trial.suggest_int("n_blocks", 1, 5),
                    d_block=trial.suggest_int("d_block", 10, 500),
                    dropout1=trial.suggest_float("dropout1", 0, 1),
                    dropout2=trial.suggest_float("dropout2", 0, 1),
                    d_hidden_multiplier=trial.suggest_float("d_hidden_multiplier", 0.5, 3),
                    learning_rate=trial.suggest_float("learning_rate", 1e-4, 5e-2, log=True),
                    weight_decay=trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
                    n_epochs=N_EPOCHS,
                    patience=PATIENCE,
                    batch_size=BATCH_SIZE,
                    seed=args.seed,
                    checkpoint_path=checkpoint_path
                )

                tr_loader, _, _ = m.prepare_data(X_train_, y_train_, X_val, y_val, None, None)
                
                epochs_run_trained = m.fit(X_train_, y_train_,X_val, y_val)
                trial.suggest_int('n_epochs', epochs_run_trained, epochs_run_trained)

                mu, sigma = m.predict_with_uncertainty(X_val, tr_loader)
                return float(np.mean(crps_gaussian(y_val, mu, sigma)))
            
            st_rmse = optuna.create_study(
                sampler=optuna.samplers.TPESampler(seed=args.seed),
                direction="minimize",
            )
            st_crps = optuna.create_study(
                sampler=optuna.samplers.TPESampler(seed=args.seed),
                direction="minimize",
            )
            st_rmse.optimize(resnet_rmse, n_trials=N_TRIALS)
            st_crps.optimize(resnet_crps, n_trials=N_TRIALS)

            for name, study, metric in [
                ("ResNet", st_rmse, "RMSE"),
                ("ResNet", st_crps, "CRPS"),
            ]:
                bp = study.best_params
                checkpoint_path = os.path.join(
                    out_dir, f"checkpoint_resnet_{metric.lower()}.pt"
                )
                m = ResNetRegressor(
                    **bp,
                    patience = PATIENCE,
                    batch_size = BATCH_SIZE,
                    seed = args.seed,
                    checkpoint_path=checkpoint_path
                )
                tr_loader, _, _ = m.prepare_data(X_train_, y_train_, None, None, None, None)

                m.fit(X_tr_p, y_tr_arr)
                if metric == "RMSE":
                    mu_test = m.predict(X_te_p)
                    val = evaluate_rmse(y_te_arr, mu_test)
                else:
                    mu_test, sigma_test = m.predict_with_uncertainty(X_te_p, tr_loader)
                    val = float(np.mean(crps_gaussian(y_te_arr, mu_test, sigma_test)))
                records.append(
                    {
                        "suite_id": args.suite_id,
                        "task_id": args.task_id,
                        "split_method": name_split,
                        "model": name,
                        "metric": metric,
                        "value": val,
                    }
                )
        else:
            def resnet_ll(trial):
                checkpoint_path = os.path.join( 
                    out_dir, f"checkpoint_resnet_ll_{trial.number}.pt"
                )
                m = ResNetClassifier(
                    n_blocks=trial.suggest_int("n_blocks", 1, 5),
                    d_block=trial.suggest_int("d_block", 10, 500),
                    dropout1=trial.suggest_float("dropout1", 0, 1),
                    dropout2=trial.suggest_float("dropout2", 0, 1),
                    d_hidden_multiplier=trial.suggest_float("d_hidden_multiplier", 0.5, 3),
                    learning_rate=trial.suggest_float("learning_rate", 1e-4, 5e-2, log=True),
                    weight_decay=trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
                    n_epochs=N_EPOCHS,
                    patience=PATIENCE,
                    batch_size=BATCH_SIZE,
                    seed=args.seed,
                    checkpoint_path=checkpoint_path
                )
                epochs_run_trained = m.fit(X_train_, y_train_, X_val, y_val)
                trial.suggest_int('n_epochs', epochs_run_trained, epochs_run_trained) 

                probs = m.predict_proba(X_val)
                return evaluate_log_loss(y_val, probs)
            
            def resnet_acc(trial):
                checkpoint_path = os.path.join( 
                    out_dir, f"checkpoint_resnet_acc_{trial.number}.pt"
                )
                m = ResNetClassifier(
                    n_blocks=trial.suggest_int("n_blocks", 1, 5),
                    d_block=trial.suggest_int("d_block", 10, 500),
                    dropout1=trial.suggest_float("dropout1", 0, 1),
                    dropout2=trial.suggest_float("dropout2", 0, 1),
                    d_hidden_multiplier=trial.suggest_float("d_hidden_multiplier", 0.5, 3),
                    learning_rate=trial.suggest_float("learning_rate", 1e-4, 5e-2, log=True),
                    weight_decay=trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
                    n_epochs=N_EPOCHS,
                    patience=PATIENCE,
                    batch_size=BATCH_SIZE,
                    seed=args.seed,
                    checkpoint_path=checkpoint_path
                )
                epochs_run_trained = m.fit(X_train_, y_train_, X_val, y_val)
                trial.suggest_int('n_epochs', epochs_run_trained, epochs_run_trained)

                preds = m.predict(X_val)
                return evaluate_accuracy(y_val, preds)
            
            st_ll = optuna.create_study(
                sampler=optuna.samplers.TPESampler(seed=args.seed),
                direction="minimize",
            )
            st_acc = optuna.create_study(
                sampler=optuna.samplers.TPESampler(seed=args.seed),
                direction="maximize",
            )
            st_ll.optimize(resnet_ll, n_trials=N_TRIALS)
            st_acc.optimize(resnet_acc, n_trials=N_TRIALS)
            for name, study, metric in [
                ("ResNet", st_ll, "LogLoss"),
                ("ResNet", st_acc, "Accuracy"),
            ]:
                bp = study.best_params
                checkpoint_path = os.path.join(
                    out_dir, f"checkpoint_resnet_{metric.lower()}.pt"
                )
                m = ResNetClassifier(
                    **bp,
                    patience = PATIENCE,
                    batch_size = BATCH_SIZE,
                    seed = args.seed,
                    checkpoint_path = checkpoint_path
                )
                m.fit(X_tr_p, y_tr_arr)
                if metric == "LogLoss":
                    probs = m.predict_proba(X_te_p)
                    val = evaluate_log_loss(y_te_arr, probs)
                else:
                    preds = m.predict(X_te_p)
                    val = evaluate_accuracy(y_te_arr, preds)
                records.append(
                    {
                        "suite_id": args.suite_id,
                        "task_id": args.task_id,
                        "split_method": name_split,
                        "model": name,
                        "metric": metric,
                        "value": val,
                    }
                )
        #FTTransformer
        if is_reg:
            def fttrans_rmse(trial):
                checkpoint_path = os.path.join(
                    out_dir, f"checkpoint_fttrans_rmse_{trial.number}.pt"
                )
                m = FTTrans_Regressor(
                    n_blocks=trial.suggest_int("n_blocks", 1, 5),
                    d_block_multiplier=trial.suggest_int("d_block_multiplier", 1, 25),
                    attention_n_heads=trial.suggest_int("attention_n_heads", 1, 20),
                    attention_dropout=trial.suggest_float("attention_dropout", 0, 1),
                    ffn_d_hidden_multiplier=trial.suggest_float("ffn_d_hidden_multiplier", 0.5, 3),
                    ffn_dropout=trial.suggest_float("ffn_dropout", 0, 1),
                    residual_dropout=trial.suggest_float("residual_dropout", 0, 1),
                    learning_rate=trial.suggest_float("learning_rate", 1e-4, 5e-2, log=True),
                    weight_decay=trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
                    n_epochs=N_EPOCHS,
                    patience=PATIENCE,
                    batch_size=BATCH_SIZE,
                    seed=args.seed,
                    checkpoint_path=checkpoint_path,
                )
                # train & record actual epochs
                epochs_trained = m.fit(X_train_, y_train_, X_val, y_val)
                trial.suggest_int("n_epochs", epochs_trained, epochs_trained)

                mu = m.predict(X_val)

                import torch, gc
                score = evaluate_rmse(y_val, mu)
                del m, mu
                torch.cuda.empty_cache()
                gc.collect()
                return score
            
            def fttrans_crps(trial):
                checkpoint_path = os.path.join(
                    out_dir, f"checkpoint_fttrans_crps_{trial.number}.pt"
                )
                m = FTTrans_Regressor(
                    n_blocks=trial.suggest_int("n_blocks", 1, 5),
                    d_block_multiplier=trial.suggest_int("d_block_multiplier", 1, 25),
                    attention_n_heads=trial.suggest_int("attention_n_heads", 1, 20),
                    attention_dropout=trial.suggest_float("attention_dropout", 0, 1),
                    ffn_d_hidden_multiplier=trial.suggest_float("ffn_d_hidden_multiplier", 0.5, 3),
                    ffn_dropout=trial.suggest_float("ffn_dropout", 0, 1),
                    residual_dropout=trial.suggest_float("residual_dropout", 0, 1),
                    learning_rate=trial.suggest_float("learning_rate", 1e-4, 5e-2, log=True),
                    weight_decay=trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
                    n_epochs=N_EPOCHS,
                    patience=PATIENCE,
                    batch_size=BATCH_SIZE,
                    seed=args.seed,
                    checkpoint_path=checkpoint_path,
                )
                tr_loader, val_loader, _ = m.prepare_data(
                    X_train_, y_train_, X_val, y_val, None, None
                )
                epochs_trained = m.fit(X_train_, y_train_, X_val, y_val)
                trial.suggest_int("n_epochs", epochs_trained, epochs_trained)

                mu, sigma = m.predict_with_uncertainty(X_val, tr_loader)

                import torch, gc
                score = float(np.mean(crps_gaussian(y_val, mu, sigma)))
                del m, tr_loader, val_loader, mu, sigma
                torch.cuda.empty_cache()
                gc.collect()
                return score
            

            study_ft_rmse = optuna.create_study(
                sampler=optuna.samplers.TPESampler(seed=args.seed),
                direction="minimize",
            )
            study_ft_crps = optuna.create_study(
                sampler=optuna.samplers.TPESampler(seed=args.seed),
                direction="minimize",
            )
            study_ft_rmse.optimize(fttrans_rmse, n_trials=N_TRIALS)
            study_ft_crps.optimize(fttrans_crps, n_trials=N_TRIALS)

            for name, study, metric in [
                ("FTTransformer", study_ft_rmse, "RMSE"),
                ("FTTransformer", study_ft_crps, "CRPS"),
            ]:
                bp = study.best_params
                checkpoint_path = os.path.join(
                    out_dir, f"checkpoint_fttrans_{metric.lower()}.pt"
                )
                m = FTTrans_Regressor(
                    **bp,
                    patience=PATIENCE,
                    batch_size=BATCH_SIZE,
                    seed=args.seed,
                    checkpoint_path=checkpoint_path,
                )
                tr_loader, _, _ = m.prepare_data(
                    X_train_, y_train_, None, None, None, None
               )
                m.fit(X_tr_p, y_tr_arr)
                if metric == "RMSE":
                    mu_test = m.predict(X_te_p)
                    val = evaluate_rmse(y_te_arr, mu_test)
                else:
                    mu_test, sigma_test = m.predict_with_uncertainty(X_te_p, tr_loader)
                    val = float(np.mean(crps_gaussian(y_te_arr, mu_test, sigma_test)))
                records.append(
                    {
                        "suite_id": args.suite_id,
                        "task_id": args.task_id,
                        "split_method": name_split,
                        "model": name,
                        "metric": metric,
                        "value": val,
                    }
                )
        else:
            def fttrans_ll(trial):
                checkpoint_path = os.path.join(
                    out_dir, f"checkpoint_fttrans_ll_{trial.number}.pt"
                )
                m = FTTrans_Classifier(
                    n_blocks=trial.suggest_int("n_blocks", 1, 5),
                    d_block_multiplier=trial.suggest_int("d_block_multiplier", 1, 25),
                    attention_n_heads=trial.suggest_int("attention_n_heads", 1, 20),
                    attention_dropout=trial.suggest_float("attention_dropout", 0, 1),
                    ffn_d_hidden_multiplier=trial.suggest_float("ffn_d_hidden_multiplier", 0.5, 3),
                    ffn_dropout=trial.suggest_float("ffn_dropout", 0, 1),
                    residual_dropout=trial.suggest_float("residual_dropout", 0, 1),
                    learning_rate=trial.suggest_float("learning_rate", 1e-4, 5e-2, log=True),
                    weight_decay=trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
                    n_epochs=N_EPOCHS,
                    patience=PATIENCE,
                    batch_size=BATCH_SIZE,
                    seed=args.seed,
                    checkpoint_path=checkpoint_path,
                )
                epochs_trained = m.fit(X_train_, y_train_, X_val, y_val)
                trial.suggest_int("n_epochs", epochs_trained, epochs_trained)

                probs = m.predict_proba(X_val)
                import torch, gc
                score = evaluate_log_loss(y_val, probs)
                del m, probs
                torch.cuda.empty_cache()
                gc.collect()
                return score
                


            def fttrans_acc(trial):
                checkpoint_path = os.path.join(
                    out_dir, f"checkpoint_fttrans_acc_{trial.number}.pt"
                )
                m = FTTrans_Classifier(
                    n_blocks=trial.suggest_int("n_blocks", 1, 5),
                    d_block_multiplier=trial.suggest_int("d_block_multiplier", 1, 25),
                    attention_n_heads=trial.suggest_int("attention_n_heads", 1, 20),
                    attention_dropout=trial.suggest_float("attention_dropout", 0, 1),
                    ffn_d_hidden_multiplier=trial.suggest_float("ffn_d_hidden_multiplier", 0.5, 3),
                    ffn_dropout=trial.suggest_float("ffn_dropout", 0, 1),
                    residual_dropout=trial.suggest_float("residual_dropout", 0, 1),
                    learning_rate=trial.suggest_float("learning_rate", 1e-4, 5e-2, log=True),
                    weight_decay=trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
                    n_epochs=N_EPOCHS,
                    patience=PATIENCE,
                    batch_size=BATCH_SIZE,
                    seed=args.seed,
                    checkpoint_path=checkpoint_path,
                )
                epochs_trained = m.fit(X_train_, y_train_, X_val, y_val)
                trial.suggest_int("n_epochs", epochs_trained, epochs_trained)

                preds = m.predict(X_val)

                import torch, gc
                score = evaluate_accuracy(y_val, preds)
                del m, preds
                torch.cuda.empty_cache()
                gc.collect()
                return score
            


            study_ft_ll = optuna.create_study(
                sampler=optuna.samplers.TPESampler(seed=args.seed),
                direction="minimize",
            )
            study_ft_acc = optuna.create_study(
                sampler=optuna.samplers.TPESampler(seed=args.seed),
                direction="maximize",
            )
            study_ft_ll.optimize(fttrans_ll, n_trials=N_TRIALS)
            study_ft_acc.optimize(fttrans_acc, n_trials=N_TRIALS)

            for name, study, metric in [
                ("FTTransformer", study_ft_ll, "LogLoss"),
                ("FTTransformer", study_ft_acc, "Accuracy"),
            ]:
                bp = study.best_params
                checkpoint_path = os.path.join(
                    out_dir, f"checkpoint_fttrans_{metric.lower()}.pt"
                )
                m = FTTrans_Classifier(
                    **bp,
                    patience=PATIENCE,
                    batch_size=BATCH_SIZE,
                    seed=args.seed,
                    checkpoint_path=checkpoint_path,
                )
                m.fit(X_tr_p, y_tr_arr)
                if metric == "LogLoss":
                    probs = m.predict_proba(X_te_p)
                    val = evaluate_log_loss(y_te_arr, probs)
                else:
                    preds = m.predict(X_te_p)
                    val = evaluate_accuracy(y_te_arr, preds)
                records.append(
                    {
                        "suite_id": args.suite_id,
                        "task_id": args.task_id,
                        "split_method": name_split,
                        "model": name,
                        "metric": metric,
                        "value": val,
                    }
                )

            


                


    pd.DataFrame(records).to_csv(out_file, index=False)
    print(f"Saved neural results to {out_file}")


if __name__ == "__main__":
    main()
