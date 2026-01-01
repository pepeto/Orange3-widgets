import copy
import sys
import numpy as np
import lightgbm as lgb
import optuna

from AnyQt.QtWidgets import (
    QGridLayout, QLabel, QSpinBox, QDoubleSpinBox,
    QCheckBox, QComboBox, QGroupBox, QVBoxLayout, QLineEdit
)
from AnyQt.QtCore import Qt

from Orange.data import Table, DiscreteVariable
from Orange.classification import Learner, Model
from Orange.regression import Learner as RegLearner, Model as RegModel
from Orange.widgets import gui
from Orange.widgets.widget import OWWidget, Input, Output
from Orange.widgets.settings import Setting
from Orange.preprocess import Preprocess
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.stats import rankdata

# Disable Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Helper for custom metrics
def get_custom_feval(metric_name, average_mode='weighted', is_multiclass=False, num_classes=None):
    def feval(preds, train_data):
        labels = train_data.get_label()
        
        if is_multiclass:
            # preds is (N * C) array, need to reshape
            preds = preds.reshape(len(labels), num_classes)
            # argmax for class labels
            pred_labels = np.argmax(preds, axis=1)
        else:
            # Binary: preds are probabilities of class 1
            pred_labels = (preds > 0.5).astype(int)
            # Binary usually ignores 'weighted/macro' but we can pass it if user wants, 
            # though usually 'binary' is the default for binary cases in sklearn.
            # We'll stick to 'binary' average for binary classification to be safe, 
            # or respect user choice if they really want 'weighted' binary?
            # Standard sklearn behavior: for binary, average='binary' is default. 
            # If user selects 'macro' in binary, it calculates macro average of both classes (0 and 1).
            pass

        # Use efficient average mode
        eff_avg_mode = average_mode
        if not is_multiclass and average_mode == 'weighted':
             # Fallback for binary if needed, but 'binary' is safer for pure pos class focus
             # However, if user wants Weighted F1 in binary (rare), we allow it.
             # But usually for binary optimization we care about the positive class (average='binary')
             # Let's override to 'binary' if mode is 'weighted' to avoid confusion, 
             # UNLESS the user explicitly selected Macro/Micro which implies they care about both classes.
             eff_avg_mode = 'binary' if average_mode == 'weighted' else average_mode

        if metric_name == 'precision':
            val = precision_score(labels, pred_labels, average=eff_avg_mode, zero_division=0)
            return 'precision', val, True
        elif metric_name == 'recall':
            val = recall_score(labels, pred_labels, average=eff_avg_mode, zero_division=0)
            return 'recall', val, True
        elif metric_name == 'f1':
            val = f1_score(labels, pred_labels, average=eff_avg_mode, zero_division=0)
        return 'f1', val, True
        
        return 'unknown', 0.0, True
    return feval


class LightGBMModel(Model):
    def __init__(self, boosters, domain, problem_type, ensemble_method='average'):
        super().__init__(domain)
        # boosters can be a single booster or a list of boosters for ensemble
        if isinstance(boosters, list):
            self.boosters = boosters
        else:
            self.boosters = [boosters]
        self.problem_type = problem_type  # 'classification' or 'regression'
        self.ensemble_method = ensemble_method  # 'average' or 'rank'
        self.used_params = {} # To store parameters used for training

    def predict(self, X):
        # Orange Model.predict expects prediction values
        # X is numpy array
        if self.problem_type == 'classification':
            # Collect predictions from all boosters
            all_preds = []
            for booster in self.boosters:
                y_pred = booster.predict(X)
                if y_pred.ndim == 1 or (y_pred.ndim == 2 and y_pred.shape[1] == 1):
                    # Binary - create (N, 2) array
                    probs = np.zeros((len(X), 2))
                    probs[:, 1] = y_pred.ravel()
                    probs[:, 0] = 1 - probs[:, 1]
                    all_preds.append(probs)
                else:
                    all_preds.append(y_pred)
            
            if self.ensemble_method == 'rank' and len(self.boosters) > 1:
                # Rank-based ensemble: convert probabilities to ranks, then average
                # This handles calibration differences between models
                all_ranks = []
                for preds in all_preds:
                    # Rank each column (class) independently
                    # Higher probability = higher rank
                    ranks = np.zeros_like(preds)
                    for col in range(preds.shape[1]):
                        ranks[:, col] = rankdata(preds[:, col], method='average')
                    all_ranks.append(ranks)
                # Average ranks
                avg_ranks = np.mean(all_ranks, axis=0)
                # Normalize to sum to 1 per row (pseudo-probabilities)
                final_probs = avg_ranks / avg_ranks.sum(axis=1, keepdims=True)
                return final_probs
            else:
                # Standard probability averaging
                return np.mean(all_preds, axis=0)
        else:
            # Regression - average predictions (rank doesn't apply well)
            all_preds = [booster.predict(X) for booster in self.boosters]
            return np.mean(all_preds, axis=0)

    def __call__(self, data, ret=Model.Value):
        # Orange abstract model call
        X = data.X if isinstance(data, Table) else data
        
        # Safe numpy conversion
        if hasattr(X, "astype"):
            X = X.astype(np.float32)
            
        probs = None
        values = None
        
        if self.problem_type == "classification":
            probs = self.predict(X)
            # Argmax for values
            values = np.argmax(probs, axis=1)
        else:
            values = self.predict(X)
            
        if ret == Model.Value:
            return values
        elif ret == Model.Probs:
            return probs
        else: # ValueProbs
            return values, probs


class LightGBMLearner(Learner):
    def __init__(self, 
                 preprocessors=None, 
                 params=None, 
                 use_optuna=False,
                 optuna_trials=10,
                 optuna_folds=3,
                 optuna_seeds=1,
                 stratified_cv=True,
                 prediction_seeds=1,
                 ensemble_method='average',
                 opt_params_flags=None,
                 opt_ranges=None,
                 name="LightGBM",
                 metric="logloss",
                 metric_avg_method="weighted",
                 balance_classes=False,
                 random_seed=42):
        super().__init__(preprocessors=preprocessors)
        self.params = params or {}
        self.use_optuna = use_optuna
        self.optuna_trials = optuna_trials
        self.optuna_folds = optuna_folds
        self.optuna_seeds = optuna_seeds
        self.stratified_cv = stratified_cv
        self.prediction_seeds = prediction_seeds
        self.ensemble_method = ensemble_method
        self.opt_params_flags = opt_params_flags or {}
        self.opt_ranges = opt_ranges or {}
        self.name = name
        self.metric = metric
        self.metric_avg_method = metric_avg_method
        self.balance_classes = balance_classes
        self.random_seed = random_seed

    def fit_storage(self, data):
        # 1. Determine problem type
        class_var = data.domain.class_var
        if class_var is None:
            raise ValueError("Data has no target variable.")

        is_classification = class_var.is_discrete
        
        X = data.X.astype(np.float32)
        y = data.Y.ravel()
        
        # Prepare LightGBM dataset
        # Identify categorical features to use LightGBM's optimal split for categories
        cat_features = [i for i, v in enumerate(data.domain.attributes) if v.is_discrete]
        
        lgb_train = lgb.Dataset(X, y, categorical_feature=cat_features if cat_features else None)

        # Base params
        final_params = copy.deepcopy(self.params)
        final_params['verbosity'] = -1
        final_params['seed'] = self.random_seed
        final_params['deterministic'] = True
        final_params['force_col_wise'] = True # Ensure determinism
        
        num_classes = 0
        if is_classification:
            num_classes = len(class_var.values)

        # Handle Class Balance
        if self.balance_classes:
            if is_classification:
                if num_classes > 2:
                    # For multiclass, using 'is_unbalance' in LightGBM is generally valid for OVA, 
                    # but for softmax it might expect class_weight. 
                    # standard 'is_unbalance' is safe for simple re-weighting
                    final_params['is_unbalance'] = True
                else:
                    final_params['is_unbalance'] = True
        
        # Determine Metric and Objective
        # metric argument: 'logloss', 'auc', 'error', 'precision', 'recall', 'f1'
        
        lgb_metric_name = "rmse" # default for regression
        is_maximize = False
        feval = None

        if is_classification:
            # num_classes already set above
            if num_classes > 2:
                final_params['objective'] = 'multiclass'
                final_params['num_class'] = num_classes
                
                # Multi metrics
                if self.metric == 'auc':
                    lgb_metric_name = 'auc_mu' # Use auc_mu for multiclass
                    is_maximize = True
                elif self.metric == 'error':
                    lgb_metric_name = 'multi_error'
                elif self.metric == 'logloss':
                    lgb_metric_name = 'multi_logloss'
                elif self.metric in ['precision', 'recall', 'f1']:
                    lgb_metric_name = self.metric # Custom name
                    is_maximize = True
                    feval = get_custom_feval(self.metric, average_mode=self.metric_avg_method, is_multiclass=True, num_classes=num_classes)
            else:
                final_params['objective'] = 'binary'
                
                # Binary metrics
                if self.metric == 'auc':
                    lgb_metric_name = 'auc'
                    is_maximize = True
                elif self.metric == 'error':
                    lgb_metric_name = 'binary_error'
                elif self.metric == 'logloss':
                    lgb_metric_name = 'binary_logloss'
                elif self.metric in ['precision', 'recall', 'f1']:
                    lgb_metric_name = self.metric
                    is_maximize = True
                    feval = get_custom_feval(self.metric, average_mode=self.metric_avg_method, is_multiclass=False)

            # If using custom eval like P/R/F1, we might not set 'metric' in params, or set 'None' to avoid built-in usage
            if feval:
                final_params['metric'] = 'None'
            else:
                final_params['metric'] = lgb_metric_name
        else:
            final_params['objective'] = 'regression'
            final_params['metric'] = 'rmse'
            lgb_metric_name = 'rmse'
            
        # Optuna Optimization
        if self.use_optuna:
            
            def objective(trial):
                # Suggest params based on flags and ranges
                trial_params = copy.deepcopy(final_params)
                
                # Helpers for safe ranges (ensure min <= max)
                def get_range(key, default_min, default_max):
                    r_min, r_max = self.opt_ranges.get(key, (default_min, default_max))
                    return min(r_min, r_max), max(r_min, r_max)

                if self.opt_params_flags.get('learning_rate'):
                    rmin, rmax = get_range('learning_rate', 1e-4, 1.0)
                    trial_params['learning_rate'] = trial.suggest_float('learning_rate', rmin, rmax, log=True)
                
                if self.opt_params_flags.get('num_leaves'):
                    rmin, rmax = get_range('num_leaves', 2, 512)
                    trial_params['num_leaves'] = trial.suggest_int('num_leaves', int(rmin), int(rmax))
                    
                if self.opt_params_flags.get('max_depth'):
                    # Handle -1 logic
                    rmin, rmax = get_range('max_depth', 3, 50)
                    
                    # We interpret the range as the CONSTRAINED range. 
                    # We still give a chance to select Unlimited (-1)
                    depth_mode = trial.suggest_categorical('depth_mode', ['unlimited', 'constrained'])
                    if depth_mode == 'unlimited':
                        trial_params['max_depth'] = -1
                    else:
                        trial_params['max_depth'] = trial.suggest_int('max_depth_int', int(rmin), int(rmax))
                    
                if self.opt_params_flags.get('n_estimators'):
                    rmin, rmax = get_range('n_estimators', 10, 5000)
                    trial_params['n_estimators'] = trial.suggest_int('n_estimators', int(rmin), int(rmax))
                
                if self.opt_params_flags.get('min_child_samples'):
                    rmin, rmax = get_range('min_child_samples', 1, 200)
                    trial_params['min_child_samples'] = trial.suggest_int('min_child_samples', int(rmin), int(rmax))

                if self.opt_params_flags.get('bagging_fraction'):
                    rmin, rmax = get_range('bagging_fraction', 0.5, 1.0)
                    trial_params['bagging_fraction'] = trial.suggest_float('bagging_fraction', rmin, rmax)
                    # Ensure bagging is applied if fraction < 1
                    if trial_params.get('bagging_freq', 0) == 0:
                        trial_params['bagging_freq'] = 1

                if self.opt_params_flags.get('bagging_freq'):
                    rmin, rmax = get_range('bagging_freq', 1, 5)
                    trial_params['bagging_freq'] = trial.suggest_int('bagging_freq', int(rmin), int(rmax))

                if self.opt_params_flags.get('feature_fraction'):
                    rmin, rmax = get_range('feature_fraction', 0.5, 1.0)
                    trial_params['feature_fraction'] = trial.suggest_float('feature_fraction', rmin, rmax)

                # Cross-validation with Multiple Seeds (Repeated K-Fold)
                # To reduce variance, we run CV multiple times with different seeds and average results
                
                metric_vals = []
                best_iters = []
                
                # Use the base random_seed to generate sub-seeds deterministically
                rng = np.random.RandomState(self.random_seed)
                seeds_to_run = rng.randint(0, 100000, size=self.optuna_seeds)
                
                for seed_idx, cv_seed in enumerate(seeds_to_run):
                    # Pruning Callback
                    class CVPruningCallback:
                        def __init__(self, trial, monitor_metric):
                            self.trial = trial
                            self.monitor_metric = monitor_metric
                        
                        def __call__(self, env):
                            for item in env.evaluation_result_list:
                                if len(item) >= 3:
                                    metric_name_item = item[1]
                                    val = item[2]
                                    if metric_name_item == self.monitor_metric or metric_name_item.endswith(self.monitor_metric):
                                        # Only report from first seed to avoid step conflicts
                                        if seed_idx == 0:
                                            self.trial.report(val, step=env.iteration)
                                            if self.trial.should_prune():
                                                raise optuna.TrialPruned()
                                        break
    
                    cv_callbacks = [lgb.early_stopping(stopping_rounds=10)]
                    if seed_idx == 0:
                        cv_callbacks.append(CVPruningCallback(trial, lgb_metric_name))
    
                    cv_kwargs = {
                        'params': trial_params,
                        'train_set': lgb_train,
                        'nfold': self.optuna_folds,
                        'seed': int(cv_seed),
                        'stratified': self.stratified_cv and is_classification,
                        'callbacks': cv_callbacks
                    }
                    if feval:
                        cv_kwargs['feval'] = feval
    
                    cv_results = lgb.cv(**cv_kwargs)
                    
                    # Find metric key
                    possible_keys = [
                        f"valid {lgb_metric_name}-mean",
                        f"{lgb_metric_name}-mean"
                    ]
                    
                    metric_key = None
                    for k in possible_keys:
                        if k in cv_results:
                            metric_key = k
                            break
                    
                    if metric_key is None:
                        keys = [k for k in cv_results.keys() if k.endswith('-mean')]
                        if keys:
                            metric_key = keys[0]
                    
                    if metric_key is None:
                        continue  # Skip failed run
    
                    vals = cv_results[metric_key]
                    if is_maximize:
                        best_idx = np.argmax(vals)
                    else:
                        best_idx = np.argmin(vals)
                    
                    metric_vals.append(vals[best_idx])
                    best_iters.append(int(best_idx + 1))
                
                # Average results across seeds
                if not metric_vals:
                    return float('-inf') if is_maximize else float('inf')
                     
                final_metric = np.mean(metric_vals)
                final_best_iter = int(np.mean(best_iters))
                
                # Save best iteration (n_estimators) to trial user attributes
                trial.set_user_attr("best_n_estimators", final_best_iter)
                        
                return final_metric

            direction = 'maximize' if is_maximize else 'minimize'
            sampler = optuna.samplers.TPESampler(seed=self.random_seed)
            # HyperbandPruner is efficient for boosting
            pruner = optuna.pruners.HyperbandPruner()
            study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
            
            # Enqueue Default Parameters
            # We map current params to trial suggestions
            default_trial_params = {}
            if self.opt_params_flags.get('learning_rate'):
                default_trial_params['learning_rate'] = final_params.get('learning_rate', 0.1)
            if self.opt_params_flags.get('num_leaves'):
                default_trial_params['num_leaves'] = final_params.get('num_leaves', 31)
            if self.opt_params_flags.get('n_estimators'):
                default_trial_params['n_estimators'] = final_params.get('n_estimators', 100)
            if self.opt_params_flags.get('min_child_samples'):
                default_trial_params['min_child_samples'] = final_params.get('min_child_samples', 20)
            if self.opt_params_flags.get('bagging_fraction'):
                default_trial_params['bagging_fraction'] = final_params.get('bagging_fraction', 0.8)
            if self.opt_params_flags.get('bagging_freq'):
                default_trial_params['bagging_freq'] = final_params.get('bagging_freq', 1)
            if self.opt_params_flags.get('feature_fraction'):
                default_trial_params['feature_fraction'] = final_params.get('feature_fraction', 0.8)
            
            if self.opt_params_flags.get('max_depth'):
                md = final_params.get('max_depth', -1)
                if md == -1:
                    default_trial_params['depth_mode'] = 'unlimited'
                else:
                    default_trial_params['depth_mode'] = 'constrained'
                    default_trial_params['max_depth_int'] = md
            
            if default_trial_params:
                study.enqueue_trial(default_trial_params)
            
            study.optimize(objective, n_trials=self.optuna_trials)
            
            # Update params with best
            # Map back complex params like depth_mode
            best_p = study.best_params
            if 'depth_mode' in best_p:
                if best_p['depth_mode'] == 'unlimited':
                    best_p['max_depth'] = -1
                else:
                    best_p['max_depth'] = best_p['max_depth_int']
                del best_p['depth_mode']
                if 'max_depth_int' in best_p: del best_p['max_depth_int']

            final_params.update(best_p)
            
            # Correct n_estimators based on early stopping in CV
            if 'best_n_estimators' in study.best_trial.user_attrs:
                final_params['n_estimators'] = study.best_trial.user_attrs['best_n_estimators']

        # Train final model(s)
        # If prediction_seeds > 1, train multiple models with different seeds for ensemble
        if feval:
            pass  # metric=None is already set, harmless

        boosters = []
        rng_pred = np.random.RandomState(self.random_seed)
        pred_seeds = rng_pred.randint(0, 100000, size=self.prediction_seeds)
        
        for pred_seed in pred_seeds:
            train_params = copy.deepcopy(final_params)
            train_params['seed'] = int(pred_seed)
            booster = lgb.train(train_params, lgb_train)
            boosters.append(booster)
        
        model = LightGBMModel(boosters, data.domain, "classification" if is_classification else "regression", 
                              ensemble_method=self.ensemble_method)
        
        # Attach used params to model for GUI display
        model.used_params = final_params
        model.used_params['ensemble_size'] = self.prediction_seeds
        return model


class OWLightGBMOptuna(OWWidget):
    name = "LightGBM Optuna"
    description = "LightGBM regressor/classifier with Optuna tuning"
    icon = "icons/GMM.svg"  # Placeholder icon
    priority = 30
    category = "Custom"
    keywords = ["lightgbm", "boosting", "optuna"]

    class Inputs:
        data = Input("Data", Table)
        preprocessor = Input("Preprocessor", Preprocess)

    class Outputs:
        learner = Output("Learner", Learner)
        model = Output("Model", Model)

    # Settings
    learner_name = Setting("LightGBM")
    
    n_estimators = Setting(100)
    learning_rate = Setting(0.1)
    max_depth = Setting(-1)
    num_leaves = Setting(31)
    min_child_samples = Setting(20)
    bagging_fraction = Setting(0.8)
    bagging_freq = Setting(1)
    feature_fraction = Setting(0.8)
    
    use_optuna = Setting(False)
    optuna_trials = Setting(20)
    optuna_folds = Setting(3)
    optuna_seeds = Setting(1)
    optimization_metric = Setting(0) # 0: LogLoss, 1: AUC, 2: Error, 3: Precision, 4: Recall, 5: F1
    metric_avg_method = Setting(0) # 0: Weighted, 1: Macro, 2: Micro
    balance_classes = Setting(False)
    stratified_cv = Setting(True)
    prediction_seeds = Setting(1)
    ensemble_method = Setting(0)  # 0: Average Probability, 1: Average Rank
    
    # Flags for which params to optimize
    opt_n_estimators = Setting(False)
    opt_learning_rate = Setting(False)
    opt_max_depth = Setting(False)
    opt_num_leaves = Setting(False)
    opt_min_child_samples = Setting(False)
    opt_bagging_fraction = Setting(False)
    opt_bagging_freq = Setting(False)
    opt_feature_fraction = Setting(False)

    # Search Ranges (Min/Max)
    s_n_est_min = Setting(10)
    s_n_est_max = Setting(5000)
    
    s_lr_min = Setting(0.0001)
    s_lr_max = Setting(1.0)
    
    s_depth_min = Setting(3)
    s_depth_max = Setting(50)
    
    s_leaves_min = Setting(2)
    s_leaves_max = Setting(512)
    
    s_child_min = Setting(1)
    s_child_max = Setting(200)
    
    s_bagging_frac_min = Setting(0.5)
    s_bagging_frac_max = Setting(1.0)
    
    s_bagging_freq_min = Setting(1)
    s_bagging_freq_max = Setting(5)
    
    s_feature_frac_min = Setting(0.5)
    s_feature_frac_max = Setting(1.0)
    
    auto_apply = Setting(True)
    random_seed = Setting(42)
    
    want_main_area = False
    
    def __init__(self):
        super().__init__()
        self.data = None
        self.preprocessor = None
        self.metric_map = {0: "logloss", 1: "auc", 2: "error", 3: "precision", 4: "recall", 5: "f1"}
        
        # -- GUI --
        
        # Name Box
        name_box = gui.widgetBox(self.controlArea, "Learner Info")
        gui.lineEdit(name_box, self, "learner_name", label="Name:", callback=self.on_change)
        
        # Params Box
        box = gui.widgetBox(self.controlArea, "Fixed Parameters (Default / Fallback)")
        
        # Estimation Button
        gui.button(box, self, "Estimate Defaults from Data", callback=self.estimate_defaults,
                   tooltip="Set optimized default parameters based on dataset size (Heuristics)")
        gui.separator(box)
        
        gui.checkBox(box, self, "balance_classes", "Balance Class Weights (Prevalence)", 
                     tooltip="Automatically adjust weights inversely proportional to class frequencies.", callback=self.on_change)
        gui.checkBox(box, self, "stratified_cv", "Stratified Cross-Validation", 
                     tooltip="Preserve class proportions in each fold (recommended for classification).", callback=self.on_change)
        gui.spin(box, self, "prediction_seeds", 1, 50, label="Prediction Ensemble (Seeds)", 
                 tooltip="Train multiple models with different seeds and average predictions to reduce variance.", callback=self.on_change)
        gui.comboBox(box, self, "ensemble_method", label="Ensemble Method",
                     items=["Average Probability", "Average Rank (Calibration-Robust)"],
                     tooltip="Average Rank converts probabilities to ranks before averaging, making the ensemble robust to calibration differences between models.",
                     callback=self.on_change)
        
        gui.spin(box, self, "n_estimators", 1, 10000, label="N Estimators", callback=self.on_change)
        gui.doubleSpin(box, self, "learning_rate", 0.000001, 1.0, step=0.0001, decimals=6, label="Learning Rate", callback=self.on_change)
        gui.spin(box, self, "num_leaves", 2, 4096, label="Num Leaves", callback=self.on_change)
        gui.spin(box, self, "max_depth", -1, 100, label="Max Depth (-1 unl.)", callback=self.on_change)
        gui.spin(box, self, "min_child_samples", 1, 500, label="Min Child Samples", callback=self.on_change)
        gui.doubleSpin(box, self, "bagging_fraction", 0.1, 1.0, step=0.05, decimals=2, label="Bagging Fraction", callback=self.on_change)
        gui.spin(box, self, "bagging_freq", 0, 10, label="Bagging Freq (0=off)", callback=self.on_change)
        gui.doubleSpin(box, self, "feature_fraction", 0.1, 1.0, step=0.05, decimals=2, label="Feature Fraction", callback=self.on_change)
        gui.spin(box, self, "random_seed", 0, 100000, label="Random Seed", callback=self.on_change)

        # Optuna Box
        opt_box = gui.widgetBox(self.controlArea, "Optimization (Optuna)")
        gui.checkBox(opt_box, self, "use_optuna", "Enable Optuna Optimization", callback=self._toggle_optuna)
        
        self.trials_spin = gui.spin(opt_box, self, "optuna_trials", 5, 200, label="Number of Trials", callback=self.on_change)
        self.folds_spin = gui.spin(opt_box, self, "optuna_folds", 2, 50, label="CV Folds", callback=self.on_change)
        self.seeds_spin = gui.spin(opt_box, self, "optuna_seeds", 1, 20, label="Repeated CV (Seeds)", callback=self.on_change)
        
        gui.comboBox(opt_box, self, "optimization_metric", label="Metric to Optimize",
                     items=["Log Loss (Minimize)", "AUC (Maximize)", "Accuracy / Error (Minimize)",
                            "Precision (Maximize)", "Recall (Maximize)", "F1 Score (Maximize)"],
                     callback=self.on_change)
        
        gui.comboBox(opt_box, self, "metric_avg_method", label="Metric Avg (Multiclass)", 
                     items=["Weighted (Prevalence)", "Macro (Equal Weight)", "Micro (Global)"],
                     callback=self.on_change)
        
        gui.separator(opt_box)
        gui.label(opt_box, self, "Optimization Ranges:")
        
        # Helper to create range rows
        def range_row(parent, label, attr_chk, attr_min, attr_max, min_val, max_val, step=1, float_spin=False):
            h = gui.hBox(parent)
            chk = gui.checkBox(h, self, attr_chk, label, callback=self.on_change)
            gui.rubber(h)
            gui.label(h, self, "Min:")
            if float_spin:
                smin = gui.doubleSpin(h, self, attr_min, min_val, max_val, step=step, callback=self.on_change)
            else:
                smin = gui.spin(h, self, attr_min, min_val, max_val, callback=self.on_change)
            
            gui.label(h, self, "Max:")
            if float_spin:
                smax = gui.doubleSpin(h, self, attr_max, min_val, max_val, step=step, callback=self.on_change)
            else:
                smax = gui.spin(h, self, attr_max, min_val, max_val, callback=self.on_change)
            
            # Store widgets to enable/disable
            return chk, smin, smax

        # N Estimators
        self.w_nest = range_row(opt_box, "N Est:", "opt_n_estimators", "s_n_est_min", "s_n_est_max", 1, 10000)
        # Learning Rate
        self.w_lr = range_row(opt_box, "L. Rate:", "opt_learning_rate", "s_lr_min", "s_lr_max", 0.0001, 1.0, step=0.001, float_spin=True)
        # Max Depth
        self.w_depth = range_row(opt_box, "Max Depth:", "opt_max_depth", "s_depth_min", "s_depth_max", 1, 100)
        # Num Leaves
        self.w_leaves = range_row(opt_box, "Num Leaves:", "opt_num_leaves", "s_leaves_min", "s_leaves_max", 2, 4096)
        # Min Child
        self.w_child = range_row(opt_box, "Min Child:", "opt_min_child_samples", "s_child_min", "s_child_max", 1, 500)
        # Bagging Fraction
        self.w_bagging_frac = range_row(opt_box, "Bagging Frac:", "opt_bagging_fraction", "s_bagging_frac_min", "s_bagging_frac_max", 0.1, 1.0, step=0.05, float_spin=True)
        # Bagging Freq
        self.w_bagging_freq = range_row(opt_box, "Bagging Freq:", "opt_bagging_freq", "s_bagging_freq_min", "s_bagging_freq_max", 0, 10)
        # Feature Fraction
        self.w_feature_frac = range_row(opt_box, "Feature Frac:", "opt_feature_fraction", "s_feature_frac_min", "s_feature_frac_max", 0.1, 1.0, step=0.05, float_spin=True)
        
        # Results Box
        res_box = gui.widgetBox(self.controlArea, "Optimization Results / Used Params")
        self.results_label = QLabel("Waiting for run...")
        self.results_label.setWordWrap(True)
        res_box.layout().addWidget(self.results_label)
        
        # Set initial UI state without triggering apply
        self._toggle_optuna(trigger=False)
        
        self.commit_box = gui.auto_commit(self.controlArea, self, "auto_apply", "Apply Automatically", commit=self.apply, callback=self.on_change)
        self.update_commit_button()

        self.apply.now()

    def update_commit_button(self):
        if not hasattr(self, "commit_box"):
            return

        if self.auto_apply:
            self.commit_box.button.setText("Apply Automatically")
        else:
            self.commit_box.button.setText("Apply")

    def estimate_defaults(self):
        """
        Estimate reasonable defaults based on dataset size.
        Heuristic:
        - Num Leaves: sqrt(N_rows). Prevents overfitting on small data.
        - Min Child Samples: N_rows / 30. Prevents overfitting.
        """
        if self.data is None:
            self.results_label.setText("No data to estimate parameters.")
            return

        n_rows = len(self.data)
        
        # 1. Num Leaves
        # sqrt(100) = 10. sqrt(1000) = 31.
        est_leaves = int(np.sqrt(n_rows))
        # Cap a bit to avoid extreme simplicity or complexity by default
        if est_leaves < 5: est_leaves = 5
        if est_leaves > 64: est_leaves = 64 # Conservative default
        self.num_leaves = est_leaves
        
        # 2. Min Child Samples
        # 100 rows -> 3. 1000 -> 33.
        est_child = int(n_rows / 30)
        if est_child < 1: est_child = 1
        if est_child > 50: est_child = 50 # Conservative cap
        self.min_child_samples = est_child
        
        # 3. Adjust Search Ranges mainly for the dependent vars
        # Correctly assign to settings directly (Settings are descriptors)
        self.s_leaves_min = 2
        # Try up to 2x or 3x the heuristic
        self.s_leaves_max = min(512, max(31, int(est_leaves * 3)))
        
        self.s_child_min = 1
        self.s_child_max = min(200, max(20, int(est_child * 3)))
        
        # Force update of labels to confirm action
        self.results_label.setText(f"Estimated defaults for {n_rows} rows:<br>" 
                                   f"Leaves: {self.num_leaves}, MinChild: {self.min_child_samples}")
        
        self.apply.deferred()

    def _toggle_optuna(self, trigger=True):
        enabled = self.use_optuna
        self.trials_spin.setEnabled(enabled)
        self.folds_spin.setEnabled(enabled)
        self.seeds_spin.setEnabled(enabled)
        
        # Enable/Disable strict rows
        def set_row_enabled(widgets, en):
            widgets[0].setEnabled(en) # Checkbox
            widgets[1].setEnabled(en) # Min
            widgets[2].setEnabled(en) # Max

        set_row_enabled(self.w_nest, enabled)
        set_row_enabled(self.w_lr, enabled)
        set_row_enabled(self.w_depth, enabled)
        set_row_enabled(self.w_leaves, enabled)
        set_row_enabled(self.w_child, enabled)
        set_row_enabled(self.w_bagging_frac, enabled)
        set_row_enabled(self.w_bagging_freq, enabled)
        set_row_enabled(self.w_feature_frac, enabled)
        
        if trigger:
            self.on_change()

    def on_change(self):
        self.update_commit_button()
        if self.auto_apply:
            self.apply.deferred()

    @Inputs.data
    def set_data(self, data):
        self.data = data
        self.apply.deferred()

    @Inputs.preprocessor
    def set_preprocessor(self, preprocessor):
        self.preprocessor = preprocessor
        self.apply.deferred()
    
    @gui.deferred
    def apply(self):
        # Create params dict for fixed values
        params = {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'num_leaves': self.num_leaves,
            'min_child_samples': self.min_child_samples,
            'bagging_fraction': self.bagging_fraction,
            'bagging_freq': self.bagging_freq,
            'feature_fraction': self.feature_fraction,
            'n_jobs': -1,
            'verbose': -1,
            'feature_pre_filter': False
        }
        
        # Define flags
        opt_flags = {
            'n_estimators': self.opt_n_estimators,
            'learning_rate': self.opt_learning_rate,
            'max_depth': self.opt_max_depth,
            'num_leaves': self.opt_num_leaves,
            'min_child_samples': self.opt_min_child_samples,
            'bagging_fraction': self.opt_bagging_fraction,
            'bagging_freq': self.opt_bagging_freq,
            'feature_fraction': self.opt_feature_fraction
        }

        # Define ranges
        opt_ranges = {
            'n_estimators': (self.s_n_est_min, self.s_n_est_max),
            'learning_rate': (self.s_lr_min, self.s_lr_max),
            'max_depth': (self.s_depth_min, self.s_depth_max),
            'num_leaves': (self.s_leaves_min, self.s_leaves_max),
            'min_child_samples': (self.s_child_min, self.s_child_max),
            'bagging_fraction': (self.s_bagging_frac_min, self.s_bagging_frac_max),
            'bagging_freq': (self.s_bagging_freq_min, self.s_bagging_freq_max),
            'feature_fraction': (self.s_feature_frac_min, self.s_feature_frac_max),
        }
        
        metric_name = self.metric_map.get(self.optimization_metric, "logloss")
        
        avg_method_map = {0: 'weighted', 1: 'macro', 2: 'micro'}
        
        # Create Learner
        learner = LightGBMLearner(
            preprocessors=[self.preprocessor] if self.preprocessor else None,
            params=params,
            use_optuna=self.use_optuna,
            optuna_trials=self.optuna_trials,
            optuna_folds=self.optuna_folds,
            optuna_seeds=self.optuna_seeds,
            stratified_cv=self.stratified_cv,
            prediction_seeds=self.prediction_seeds,
            ensemble_method='average' if self.ensemble_method == 0 else 'rank',
            opt_params_flags=opt_flags,
            opt_ranges=opt_ranges,
            name=self.learner_name if self.learner_name else "LightGBM",
            metric=metric_name,
            metric_avg_method=avg_method_map.get(self.metric_avg_method, 'weighted'),
            balance_classes=self.balance_classes,
            random_seed=self.random_seed
        )
        
        # Send Learner
        self.Outputs.learner.send(learner)
        
        # If data is available, train and send model
        if self.data:
            self.error()
            self.results_label.setText("Training...")
            try:
                model = learner.fit_storage(self.data)
                self.Outputs.model.send(model)
                
                # Update Results info
                used = getattr(model, 'used_params', params)
                txt = "<b>Used Parameters:</b><br>"
                # Highlight optimized ones?
                keys = ['n_estimators', 'learning_rate', 'num_leaves', 'max_depth', 'min_child_samples', 
                        'bagging_fraction', 'bagging_freq', 'feature_fraction']
                for k in keys:
                     val = used.get(k, 'N/A')
                     if isinstance(val, float): val = f"{val:.4f}"
                     txt += f"{k}: <b>{val}</b><br>"
                
                if self.prediction_seeds > 1:
                    txt += f"ensemble_size: <b>{self.prediction_seeds}</b><br>"
                    txt += f"ensemble_method: <b>{'Rank' if self.ensemble_method == 1 else 'Average'}</b><br>"
                
                if self.use_optuna:
                    txt += f"<br><i>(Best of {self.optuna_trials} trials)</i>"
                
                self.results_label.setText(txt)
                
            except Exception as e:
                self.error(str(e))
                self.results_label.setText(f"Error: {str(e)}")
                self.Outputs.model.send(None)
        else:
            self.results_label.setText("No Data")
            self.Outputs.model.send(None)
