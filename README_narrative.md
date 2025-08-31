# Machine Learning Model Comparison for Extrapolation: A Master's Thesis Project

## Project Overview
This research project investigates the performance of various machine learning models under different extrapolation scenarios. The study systematically evaluates how different model architectures handle data points outside their training distribution, providing insights into their generalization capabilities in real-world applications where models often encounter novel data patterns.

## Research Context
In many practical machine learning applications, models are frequently required to make predictions on data that extends beyond the distribution of their training data. This project explores this challenging aspect of machine learning by implementing and comparing multiple model families under carefully designed extrapolation conditions. The research includes comprehensive hyperparameter optimization using Optuna to ensure fair comparisons across different model architectures.

## Implementation Details

The project is organized into several key components:

### Core Components
- **Model Implementations**: The repository contains specialized wrappers for various model families, including neural networks, tree-based models, and advanced architectures like TabPFN (Tabular Prior-data Fitted Networks).
- **Extrapolation Framework**: We've developed a flexible system for defining and applying different extrapolation methods, including spatial depth-based approaches, to systematically test model behavior.
- **Evaluation Suite**: A comprehensive set of metrics including accuracy, RMSE, log loss, and CRPS (Continuous Ranked Probability Score) provides a multi-faceted view of model performance.

### Technical Infrastructure
- **Data Handling**: The `loader.py` module manages data loading and preprocessing, with special handling for Tabzilla datasets.
- **Model Training**: The system supports both standard training procedures and advanced techniques like curriculum learning for extrapolation.
- **Hyperparameter Optimization**: Optuna is integrated for automated hyperparameter search, with support for parallel optimization and early stopping.

## Getting Started

### Prerequisites
- Python 3.7 or higher
- Core scientific computing stack (NumPy, SciPy, Pandas)
- PyTorch for neural network implementations
- scikit-learn for traditional ML models
- Optuna for hyperparameter optimization
- Visualization libraries (Matplotlib, Seaborn)

### Installation
1. Clone the repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running Experiments

#### On Euler Cluster
To run experiments on the Euler cluster, use the following command format:
```bash
sbatch --wrap="launcher_script.sh --task_id <task_id> --suit_id <suit_id>"
```

#### Local Execution
For local execution, you can run the main experiment scripts directly:

1. **Advanced Models**:
   ```bash
   python adv_trial.py --task_id <task_id> --suit_id <suit_id>
   ```

2. **Baseline Models**:
   ```bash
   python baseline_experiment.py --task_id <task_id> --suit_id <suit_id>
   ```

3. **Neural Network Models**:
   ```bash
   python neural_experiment.py --task_id <task_id> --suit_id <suit_id>
   ```

4. **TabPFN Models**:
   ```bash
   python TabPFN_experiment.py --task_id <task_id> --suit_id <suit_id>
   ```

5. **Tree-based Models**:
   ```bash
   python tree_experiment.py --task_id <task_id> --suit_id <suit_id>
   ```

For Tabzilla datasets, append `_tabz` to the script names (e.g., `neural_experiment_tabz.py`).

### Configuration
All configurations can be managed through the `requirements.txt` file and environment-specific settings. The hyperparameter optimization is integrated within each script, so no separate optimization step is required.

## Research Findings
Preliminary results are available in the `analysis/` directory, with visualizations organized by metric type. The findings provide insights into:
- Relative performance of different model families under extrapolation
- Impact of various extrapolation definitions on model performance
- Effectiveness of different hyperparameter optimization strategies

## Extending the Research
Researchers interested in building upon this work can:
1. Add new model implementations to the `models/` directory
2. Define new extrapolation methods in `extrapolation_methods.py`
3. Contribute additional evaluation metrics to `evaluation_metrics.py`

## License
[Specify your preferred license here]

## Contact
For questions or collaborations, please contact [Your Contact Information]
