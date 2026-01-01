# LightGBM Optuna Widget for Orange3

## Overview
The **LightGBM Optuna** widget is an advanced machine learning component for Orange3 that integrates the high-performance **LightGBM** gradient boosting framework with the state-of-the-art **Optuna** hyperparameter optimization engine. 

Unlike standard boosting implementations, this widget is designed with a heavy focus on **Data Science rigor**, specifically addressing common issues like overfitting, model variance, and calibration bias.

---

## Key Features

### 1. Optuna Hyperparameter Optimization
The widget uses Optuna to automatically search for the best configuration of parameters. 
- **Automated Search**: It optimizes Learning Rate, Number of Leaves, Max Depth, and more.
- **Pruning (Hyperband)**: Uses the Hyperband algorithm to stop unpromising trials early, significantly speeding up the optimization process.
- **Default Enqueueing**: Your manual settings are always tested as the first trial, ensuring the optimizer only provides a better (or equal) solution than your initial hand-tuned model.

### 2. Robust Cross-Validation (Repeated K-Fold)
To prevent "overfitting to the validation set," the widget implements **Repeated CV**:
- **Variance Reduction**: It runs the Cross-Validation process multiple times with different random seeds and averages the results.
- **Stability**: This ensures that the chosen parameters are robust across different data partitions, not just lucky on one specific split.
- **Stratification**: Optionally preserves class proportions in each fold, which is critical for imbalanced classification tasks.

### 3. Prediction Ensemble (Multi-Seed Training)
A single model can be sensitive to the initial random seed. To combat this, the widget supports **Multi-Seed Ensembles**:
- **Mechanism**: Trains $N$ models with identical hyperparameters but different random seeds.
- **Ensemble Averaging**: Combines the outputs of all models to produce a more stable prediction with lower variance.

### 4. Calibration-Robust Ranking (Rank Averaging)
In classification, different models might produce probabilities that are not on the same scale (calibration issues).
- **Average Probability**: The traditional method.
- **Average Rank**: Converts each model's probabilities into internal ranks before averaging. This ensures that a sample consensus on "which observation is the most likely positive" is preserved, even if the absolute probability values differ across seeds.

### 5. Native Categorical & Imbalance Handling
- **Categorical Features**: Automatically detects discrete variables in Orange and passes them to LightGBM's native categorical split algorithm (much more efficient than One-Hot Encoding).
- **Class Balancing**: Uses the `is_unbalance` logic to automatically weight classes inversely proportional to their frequency, improving performance on minority classes.

---

## Detailed Parameter Guide

### Fixed Parameters (Manual/Fallback)
These settings are used when Optuna is disabled, or as the starting point for optimization:
- **N Estimators**: Maximum number of trees. Note that if Optuna is used, this is automatically corrected by the *Best Iteration* found during CV.
- **Learning Rate**: Step size shrinkage. Now supports up to 6 decimal places for ultra-fine tuning.
- **Bagging Fraction & Feature Fraction**: Controls the percentage of data/features used per tree. Setting these below 1.0 is essential for creating diversity in sub-models.
- **Bagging Freq**: Frequency for performing bagging.

### Optimization (Optuna)
- **Number of Trials**: Total combinations to test.
- **Metric to Optimize**: Choose from LogLoss, AUC, F1, Precision, Recall, or Accuracy.
- **Metric Avg (Multiclass)**: Defines how metrics are aggregated in multiclass problems (Weighted, Macro, or Micro).

### Search Ranges
You can define strict Min and Max boundaries for the optimizer. This prevents the search from wandering into irrelevant parameter spaces and ensures the model stays within your desired complexity limits.

---

## Scientific Rationale: Why these options?

1. **Why Bagging/Feature Fraction?** Without these, different random seeds produce identical trees in simple datasets. By introducing sub-sampling, the seeds create different "views" of the data, making the ensemble effective.
2. **Why Rank Averaging?** In financial or medical models, the *relative order* of risk is often more important than the exact probability. Rank averaging captures this order robustly across multiple models.
3. **Why Best Iteration capture?** A common mistake in Optuna implementations is taking the best parameters but using the maximum `n_estimators`. This widget captures the exact tree count where the validation error was lowest to prevent late-stage overfitting.

---

## Usage Tips
- **Start with "Estimate Defaults"**: Click this button to get a heuristic setup based on your dataset size ($\sqrt{N_{rows}}$ logic).
- **Use Repeated CV for small data**: If your dataset is small, set *Repeated CV (Seeds)* to 3 or 5 to get a truly reliable optimization metric.
- **Enable Rank Averaging for sensitive ranking**: If you are building a ranking system (e.g., Lead Scoring), *Average Rank* is usually more robust than *Average Probability*.
