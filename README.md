# Documentation: Custom Orange3 Widgets

This document details the functionality, design choices, and usage guidelines for the custom widgets suite.

---

## 1. LightGBM Optuna Widget for Orange3

### Overview
Unlike standard boosting implementations, this widget is designed with a heavy focus on **Data Science rigor**, specifically addressing common issues like overfitting, model variance, and calibration bias.

---

### Key Features

#### 1. Optuna Hyperparameter Optimization
The widget uses Optuna to automatically search for the best configuration of parameters. 
- **Automated Search**: It optimizes Learning Rate, Number of Leaves, Max Depth, and more.
- **Pruning (Hyperband)**: Uses the Hyperband algorithm to stop unpromising trials early, significantly speeding up the optimization process.
- **Default Enqueueing**: Your manual settings are always tested as the first trial, ensuring the optimizer only provides a better (or equal) solution than your initial hand-tuned model.

#### 2. Robust Cross-Validation (Repeated K-Fold)
To prevent "overfitting to the validation set," the widget implements **Repeated CV**:
- **Variance Reduction**: It runs the Cross-Validation process multiple times with different random seeds and averages the results.
- **Stability**: This ensures that the chosen parameters are robust across different data partitions, not just lucky on one specific split.
- **Stratification**: Optionally preserves class proportions in each fold, which is critical for imbalanced classification tasks.

#### 3. Prediction Ensemble (Multi-Seed Training)
A single model can be sensitive to the initial random seed. To combat this, the widget supports **Multi-Seed Ensembles**:
- **Mechanism**: Trains $N$ models with identical hyperparameters but different random seeds.
- **Ensemble Averaging**: Combines the outputs of all models to produce a more stable prediction with lower variance.

#### 4. Calibration-Robust Ranking (Rank Averaging)
In classification, different models might produce probabilities that are not on the same scale (calibration issues).
- **Average Probability**: The traditional method.
- **Average Rank**: Converts each model's probabilities into internal ranks before averaging. This ensures that a sample consensus on "which observation is the most likely positive" is preserved, even if the absolute probability values differ across seeds.

#### 5. Native Categorical & Imbalance Handling
- **Categorical Features**: Automatically detects discrete variables in Orange and passes them to LightGBM's native categorical split algorithm (much more efficient than One-Hot Encoding).
- **Class Balancing**: Uses the `is_unbalance` logic to automatically weight classes inversely proportional to their frequency, improving performance on minority classes.

---

### Detailed Parameter Guide

#### Fixed Parameters (Manual/Fallback)
These settings are used when Optuna is disabled, or as the starting point for optimization:
- **N Estimators**: Maximum number of trees. Note that if Optuna is used, this is automatically corrected by the *Best Iteration* found during CV.
- **Learning Rate**: Step size shrinkage. Now supports up to 6 decimal places for ultra-fine tuning.
- **Bagging Fraction & Feature Fraction**: Controls the percentage of data/features used per tree. Setting these below 1.0 is essential for creating diversity in sub-models.
- **Bagging Freq**: Frequency for performing bagging.

#### Optimization (Optuna)
- **Number of Trials**: Total combinations to test.
- **Metric to Optimize**: Choose from LogLoss, AUC, F1, Precision, Recall, or Accuracy.
- **Metric Avg (Multiclass)**: Defines how metrics are aggregated in multiclass problems (Weighted, Macro, or Micro).

#### Search Ranges
You can define strict Min and Max boundaries for the optimizer. This prevents the search from wandering into irrelevant parameter spaces and ensures the model stays within your desired complexity limits.

---

### Scientific Rationale: Why these options?

1. **Why Bagging/Feature Fraction?** Without these, different random seeds produce identical trees in simple datasets. By introducing sub-sampling, the seeds create different "views" of the data, making the ensemble effective.
2. **Why Rank Averaging?** In financial or medical models, the *relative order* of risk is often more important than the exact probability. Rank averaging captures this order robustly across multiple models.
3. **Why Best Iteration capture?** A common mistake in Optuna implementations is taking the best parameters but using the maximum `n_estimators`. This widget captures the exact tree count where the validation error was lowest to prevent late-stage overfitting.

---

### Usage Tips
- **Start with "Estimate Defaults"**: Click this button to get a heuristic setup based on your dataset size ($\sqrt{N_{rows}}$ logic).
- **Use Repeated CV for small data**: If your dataset is small, set *Repeated CV (Seeds)* to 3 or 5 to get a truly reliable optimization metric.
- **Enable Rank Averaging for sensitive ranking**: If you are building a ranking system (e.g., Lead Scoring), *Average Rank* is usually more robust than *Average Probability*.

---

## 2. GMM Clustering Widget (Gaussian Mixture Models)

### Overview
This widget implements **Gaussian Mixture Models (GMM)** and **Bayesian Gaussian Mixture Models (BGMM)** for clustering. Unlike standard K-Means, GMM assigns data points to clusters probabilistically (soft clustering), allowing for elliptical cluster shapes.

### Key Features
*   **Hungarian Mapping**: If ground-truth class labels are present, the widget calculates the optimal 1-to-1 mapping between discovered clusters and true classes using the Hungarian algorithm (Linear Sum Assignment). This transforms the unsupervised clustering task into a supervised classification evaluation automatically.
*   **Auto-Tune**: Can automatically select the optimal number of components ($k$) by minimizing **BIC** (Bayesian Information Criterion) or **AIC**.
*   **Bayesian GMM**: Supports Dirichlet Process priors to infer the effective number of active components automatically from a larger upper bound.
*   **Covariance Control**: Allows selection of covariance types (`full`, `tied`, `diag`, `spherical`) to constrain cluster shapes.

### Usage Logic
1.  **Scaling**: By default, applies `StandardScaler` internally. This is crucial for GMM convergence.
2.  **Mapping**: Enable "Optimal cluster-to-class mapping" only if your data typically comes with ground truth (e.g., for testing clustering capability on labeled data).
3.  **Auto-tune vs Fixed**: Use auto-tune to explore detailed structures; use fixed $k$ when you have domain knowledge (e.g., 2 market regimes).

---

## 3. HMM Clustering Widget (Hidden Markov Models)

### Overview
The HMM widget applies **Hidden Markov Models** to time-series or sequential data. It assumes that the observed data is generated by a sequence of hidden states that transition between each other with certain probabilities.

### Key Features
*   **Temporal Context**: Unlike GMM or K-Means, HMM respects the order of data rows. It groups "periods" of time into states (e.g., "Bull Market" vs "Bear Market").
*   **Multithreaded Training**: Training HMMs can be computationally expensive. This widget runs the Expectation-Maximization (EM) algorithm in a **background thread**, preventing the UI from freezing.
*   **Visual Output**: Annotates the input data with `HMM_state` (the most likely hidden state) and `HMM_confidence` (probability of that state).

### Best Practices
*   **Sort your data**: Ensure input data is strictly time-ordered before feeding it to this widget.
*   **Stationarity**: HMMs assume transition matrices are constant. If market dynamics change structurally over decades, a single HMM might struggle.
*   **States**: Start with 2 or 3 states for financial data (e.g., Low Vol/Up, High Vol/Down).

---

## 4. ZigZag & Trend Widget

### Overview
A technical analysis widget that filters noise from time-series price data to identify significant "Swings" or trends.

### Key Features
*   **ZigZag Algorithm**: Identifies local peaks and valleys that deviate from the previous extreme by at least a specified **Swing Percentage**.
*   **Trend Identification**: Generates a `Trend` column (+1 for Uptrend, -1 for Downtrend) based on the current swing direction.
*   **Noise Filtering**: Ignores moves smaller than `swing %`, making it ideal for labeling training data for trading algorithms (e.g., "classify this bar as part of an Uptrend").

### Usage
- **Input**: Connect a dataset with a Price column (Close, High, Low, etc.).
- **Parameters**: 
    - **Swing %**: The threshold for a reversal. Larger values (e.g., 5-10%) capture major trends; smaller values (1-2%) capture minor corrections.
- **Output**: Returns the original data augmented with `ZigZag` (pivot points) and `Trend` features.

---

## 5. Yahoo Finance Widget

### Overview
A data ingestion utility that downloads full historical OHLCV (Open, High, Low, Close, Volume) data directly from Yahoo Finance.

### Key Features
*   **Threaded Download**: Downloads run asynchronously, ensuring the application remains responsive even when fetching decades of daily data.
*   **Robust Parsing**: Implements "Case-Insensitive" and "Structure-Agnostic" parsing to handle variations in the `yfinance` API response (e.g., changing from MultiIndex to flat columns).
*   **Automatic Formatting**: Converts date columns and ensures numeric types are ready for Orange pipelines immediately.

