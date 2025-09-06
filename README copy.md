
## Overview

This repository contains a comprehensive evaluation framework for machine learning extrapolation methods, comparing various data splitting techniques and model performance across different experimental setups. The project systematically evaluates how different extrapolation strategies affect model generalization on tabular datasets.

## Key Features

- **Multiple Extrapolation Methods**: Implementation of 7+ splitting strategies including Mahalanobis distance, UMAP, K-means, Gower distance, K-medoids, and spatial depth-based splits
- **Comprehensive Model Suite**: Evaluation across baseline models, neural networks, tree-based methods, and TabPFN
- **Systematic Experimentation**: Structured experimental pipeline with baseline, advanced, and specialized trials
- **Robust Evaluation**: Multiple metrics and statistical analysis for reliable performance assessment
- **Scalable Architecture**: Modular design supporting easy extension with new methods and models

## Installation


### Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd <project-directory>

# Install dependencies
pip install -r requirements.txt
```


## Project Structure

```
├── src/                          # Core source code
│   ├── models/                   # Model implementations
│   │   ├── Baseline_models.py    # Traditional ML models
│   │   ├── Neural_models.py      # Deep learning models
│   │   ├── Tree_models.py        # Tree-based models
│   │   └── TabPFN_model.py       # TabPFN integration
│   ├── extrapolation_methods.py  # Data splitting strategies
│   ├── evaluation_metrics.py     # Performance metrics
│   ├── loader.py                 # Data loading utilities
│   └── utils.py                  # Helper functions
├── baseline_experiment.py        # Baseline model evaluation
├── neural_experiment.py          # Neural network experiments
├── tree_experiment_debug.py      # Tree model experiments
├── TabPFN_experiment.py          # TabPFN evaluation
├── adv_trial.py                  # Advanced experimental trials
├── launcher_*.py                 # Experiment launchers
└── requirements.txt              # Dependencies
```

## Usage

### Quick Start

1. **Run Baseline Experiments**:
```bash
python baseline_experiment.py --dataset_id <openml_dataset_id> --split_method random
```

2. **Neural Network Experiments**:
```bash
python neural_experiment.py --dataset_id <dataset_id> --split_method mahalanobis
```

3. **TabPFN Evaluation**:
```bash
python TabPFN_experiment.py --dataset_id <dataset_id>
```

### Batch Experiments

Use launcher scripts for systematic evaluation:
```bash
python launcher_baseline.py  # Run baseline across multiple datasets
python launcher_neural.py    # Neural network batch experiments
python launcher_tabpfn.py    # TabPFN batch evaluation
```

### Available Splitting Methods

- `random` - Random train/test split
- `mahalanobis` - Mahalanobis distance-based split
- `umap` - UMAP embedding-based split
- `kmeans` - K-means clustering split
- `gower` - Gower distance split
- `kmedoids` - K-medoids clustering split
- `spatial_depth` - Spatial depth-based split

### Supported Models

**Baseline Models**:
- Linear Regression
- Random Forest
- Support Vector Machines
- Gradient Boosting (LightGBM)

**Neural Models**:
- Multi-layer Perceptrons
- Residual Networks
- Attention-based architectures

**Specialized Models**:
- TabPFN (Tabular Prior-Fitted Networks)
- Tree-based ensembles
- Generalized Additive Models (GAM)

## Experimental Design

The framework supports three main experimental paradigms:

1. **Baseline Experiments** (`baseline_experiment.py`): Systematic evaluation of traditional ML methods across different splitting strategies

2. **Neural Experiments** (`neural_experiment.py`): Deep learning model evaluation with advanced architectures and hyperparameter optimization

3. **Advanced Trials** (`adv_trial.py`): Specialized experiments combining multiple techniques and advanced evaluation metrics

## Results and Analysis


Each result includes:
- Performance metrics (accuracy, F1, AUC, etc.)
- Statistical significance tests
- Computational time measurements
- Model-specific diagnostics

## Datasets

The framework evaluates models on standardized benchmark suites to ensure comprehensive and fair comparison:

### OpenML Benchmark Suites
The project primarily uses OpenML benchmark suites for systematic evaluation:

- **Suite 334**: Regression tasks with numerical features only
- **Suite 335**: Regression tasks with numerical and categorical features
- **Suite 336**: Classification tasks with numerical features only  
- **Suite 337**: Classification tasks with numerical and categorical features

### Additional Benchmarks
- **TabZilla Classification Benchmark**: Extended classification dataset collection for robust evaluation

### Usage Examples
```bash
python baseline_experiment.py --dataset_id 31    # Credit-g dataset
python baseline_experiment.py --dataset_id 1590  # Adult dataset

# Run on specific OpenML suite
python launcher_baseline.py --suite_id 334  # Numerical regression tasks
python launcher_baseline.py --suite_id 337  # Mixed-type classification tasks
```

### Dataset Characteristics
The benchmark suites cover diverse scenarios:
- **Task Types**: Both regression and classification problems
- **Feature Types**: Numerical-only and mixed (numerical + categorical) datasets
- **Dataset Sizes**: Ranging from small (hundreds of samples) to large (tens of thousands)
- **Complexity**: Various levels of feature interactions and non-linearity

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-method`)
3. Implement your changes following the existing code structure
4. Add tests and documentation
5. Submit a pull request

### Adding New Methods

To add a new extrapolation method:
1. Implement the method in `src/extrapolation_methods.py`
2. Add corresponding tests
3. Update the experimental scripts to include the new method

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{extrapolation-methods-eval,
  title={Machine Learning Extrapolation Methods: A Comprehensive Evaluation Framework},
  author={[Your Name]},
  year={2024},
  url={[Your Repository URL]}
}
```



**Note**: This is a research project. Results may vary depending on hardware, software versions, and random seeds. For reproducible results, ensure consistent environment setup and use fixed random seeds where appropriate.
