# Machine Learning Model Comparison under Extrapolation

## Overview
This repository contains the official codebase for the thesis *Machine Learning Model Comparison under Extrapolation*.  
The project provides a large-scale, systematic benchmark of tabular learning models under **interpolation (i.i.d.)** and **extrapolation (out-of-distribution)** regimes.  

We evaluate a broad spectrum of models—ranging from simple baselines to foundation models—across nearly 100 OpenML tasks and the **TabZilla benchmark suite (ID 379)**. The framework implements multiple definitions of extrapolation sets, supports both regression and classification tasks, and reports results across complementary metrics.

Our goal is to establish a reproducible foundation for evaluating how well models generalize **beyond the training distribution**.

---

## Key Features
- **Multiple Extrapolation Strategies**  
  - Random split (baseline)  
  - Distance-based: Mahalanobis, Gower  
  - Clustering-based: k-means, k-medoids  
  - Manifold-based: UMAP  
  - Depth-based: Spatial depth  

- **Model Families**  
  - **Baselines**: constant predictor, linear regression, logistic regression  
  - **Tree ensembles**: Random Forest, LightGBM  
  - **Neural networks**: MLP, ResNet, FT-Transformer  
  - **Advanced methods**: Engression, GP, Distributional Random Forest, LightGBMLSS 
  - **Foundation model**: TabPFN  

- **Diverse Benchmarks**  
  - OpenML benchmark suites (IDs 334–337) for regression and classification (numerical / mixed)  
  - **TabZilla benchmark suite (ID 379)**: 36 diverse, deliberately difficult classification datasets  

- **Comprehensive Metrics**  
  - Point prediction: RMSE, Accuracy  
  - Distributional: CRPS  
  - Probabilistic calibration: LogLoss  
  - Degradation analysis from interpolation to extrapolation  

- **Extensible Framework**  
  Modular design allows researchers to easily add new models, extrapolation methods, or evaluation metrics.

---

## Workflow Overview

1. **Select a benchmark suite**  
   - `regression_numerical` (OpenML 336)  
   - `classification_numerical` (OpenML 337)  
   - `regression_numerical_categorical` (OpenML 335)  
   - `classification_numerical_categorical` (OpenML 334)  
   - `tabzilla` (OpenML 379)  

2. **Choose an experiment type**  
   - `baseline_experiment.py` — linear/logistic/constant models  
   - `neural_experiment.py` — MLP, ResNet, FT-Transformer  
   - `tree_experiment.py` — Random Forest, LightGBM  
   - `adv_trial.py` — Engression, GPBoost  
   - `TabPFN_experiment.py` — TabPFN foundation model  

3. **Apply extrapolation splits**  
   The framework automatically selects compatible splitters depending on dataset type:  
   - **Numerical datasets** → random, Mahalanobis, k-means, UMAP, spatial depth  
   - **Mixed numerical + categorical datasets** → random, Gower, k-medoids, UMAP  

   Categorical variables are encoded as category codes for splitters that require numeric input.

4. **Run experiments and collect results**  
   Metrics, plots, and LaTeX tables are saved in the `analysis/` folder.  
   These outputs directly generated the figures and tables in the accompanying thesis.

---

## TabZilla Experiments

The **TabZilla benchmark suite (ID 379)** provides a deliberately hard and diverse set of classification datasets.  
To ensure consistency, every experiment script has a dedicated TabZilla counterpart:

- `baseline_experiment_tabz.py`  
- `neural_experiment_tabz.py`  
- `tree_experiment_tabz.py`  
- `adv_trial_tabz.py`  
- `TabPFN_experiment_tabz.py`  

These scripts follow the same workflow as the standard experiments but are adapted for TabZilla’s heterogeneous datasets. This duplication makes it explicit which runs correspond to standard OpenML suites and which to TabZilla.

---

## Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/alicestratula/master_thesis
cd <project-directory>
pip install -r requirements.txt
