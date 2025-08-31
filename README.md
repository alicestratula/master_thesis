# Machine Learning Model Comparison for Extrapolation

This repository contains the implementation and analysis for a master's thesis project comparing various machine learning models under different extrapolation scenarios. The project focuses on evaluating model performance across various extrapolation definitions and includes comprehensive hyperparameter optimization using Optuna.

## Project Structure

```
master_thesis/
├── analysis/                  # Analysis results and visualizations
│   └── PICTURES/              # Generated plots
│       ├── accuracy/          # Accuracy-related visualizations
│       ├── crps/              # CRPS (Continuous Ranked Probability Score) plots
│       └── logloss/           # Log loss visualizations
│
└── src/                       # Source code
    ├── models/                # Model implementations
    │   ├── Advanced_models.py # Advanced model architectures
    │   ├── Baseline_models.py # Baseline model implementations
    │   ├── Neural_models.py   # Neural network implementations
    │   ├── Tree_models.py     # Tree-based models
    │   └── TabPFN_model.py    # TabPFN model implementation
    │
    ├── evaluation_metrics.py  # Custom evaluation metrics
    ├── extrapolation_methods.py # Different extrapolation techniques
    ├── loader.py              # Data loading and preprocessing
    ├── utils.py               # Utility functions
    └── config_defaults        # Default configuration parameters
```

## Features

- **Model Wrappers**: Implementations for various model families including:
  - Neural Networks
  - Tree-based models
  - Baseline models
  - TabPFN (Tabular Prior-data Fitted Networks)
  - Advanced model architectures

- **Extrapolation Methods**:
  - Multiple extrapolation definitions
  - Spatial depth-based methods
  - Dataset-specific adaptations

- **Hyperparameter Optimization**:
  - Optuna integration for automated hyperparameter tuning
  - Configurable search spaces
  - Parallel optimization support

- **Evaluation**:
  - Comprehensive metrics (Accuracy, RMSE, Log Loss, CRPS)
  - Visualization tools
  - Statistical analysis

## Requirements

- Python 3.7+
- PyTorch
- scikit-learn
- Optuna
- NumPy
- Pandas
- Matplotlib/Seaborn for visualization

## Usage

1. **Data Preparation**:
   - Place your datasets in the appropriate directory
   - Configure the data loading parameters in `loader.py`

2. **Model Training**:
   ```bash
   python src/launchers/train.py --model <model_name> --dataset <dataset_name>
   ```

3. **Hyperparameter Optimization**:
   ```bash
   python src/launchers/optimize.py --model <model_name> --dataset <dataset_name>
   ```

4. **Evaluation**:
   ```bash
   python src/launchers/evaluate.py --model <model_name> --dataset <dataset_name>
   ```

## Tabzilla Integration

The project includes adaptations for Tabzilla datasets (marked with `_tabz` suffix in filenames) which require additional handling.

## Results

Analysis results and visualizations are stored in the `analysis/` directory, including:
- Accuracy comparisons
- RMSE (Root Mean Square Error) plots
- Log Loss metrics
- CRPS (Continuous Ranked Probability Score) analysis

## License

[Specify your license here]

## Acknowledgments

[Any acknowledgments or references you'd like to include]
