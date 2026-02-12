import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
import sys

# Scikit-learn Models & Utils
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedShuffleSplit, ParameterGrid
from sklearn.metrics import (roc_curve, auc, roc_auc_score, accuracy_score, 
                             precision_score, recall_score, f1_score)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==========================================
# Configuration & Path Settings
# ==========================================
# NOTE: Update these paths before running
BASE_PATH = './'  # Current directory or absolute path
DATA_DIR = os.path.join(BASE_PATH, 'dataset/ML_dataset/discovery/')
VAL_DIR = os.path.join(BASE_PATH, 'dataset/ML_dataset/validation/')
LABEL_DIR = os.path.join(BASE_PATH, 'dataset/ML_dataset/')
SAVE_BASE_PATH = os.path.join(BASE_PATH, 'results/')

# Analysis Parameters
N_BOOTSTRAPS = 1000
RNG_SEED = 42
RNG = np.random.RandomState(RNG_SEED)
MEAN_FPR = np.linspace(0, 1, 100)

# ==========================================
# Model Initialization Helper
# ==========================================
def get_model_config(model_name, y_train):
    """
    Returns the model instance and parameter grid based on the model name.
    """
    # Calculate scale_pos_weight for XGBoost (handling class imbalance)
    scale_pos_weight = float(np.sum(y_train == 0)) / np.sum(y_train == 1) if len(y_train) > 0 else 1.0

    if model_name == 'svm':
        model = svm.SVC(probability=True, random_state=RNG_SEED, class_weight='balanced')
        param_grid = [
            {'C': [2 ** s for s in range(-5, 6, 2)], 'kernel': ['linear']},
            {'C': [2 ** s for s in range(-3, 5, 2)], 'gamma': [2 ** s for s in range(-3, -13, -2)], 'kernel': ['rbf']}
        ]
        
    elif model_name == 'rf':
        model = RandomForestClassifier(random_state=RNG_SEED, class_weight='balanced')
        param_grid = {
            'n_estimators': [100, 300],
            'max_depth': [5, 10, None],
            'min_samples_leaf': [1, 2]
        }
        
    elif model_name == 'knn':
        model = KNeighborsClassifier()
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }
        
    elif model_name == 'xgboost':
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RNG_SEED)
        param_grid = {
            'n_estimators': [100, 300],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'scale_pos_weight': [1, scale_pos_weight]
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model, list(ParameterGrid(param_grid))

# ==========================================
# Main Analysis Pipeline
# ==========================================
def run_analysis(target_model_name):
    print(f"🔹 Starting Analysis for Model: {target_model_name.upper()}")
    
    # Create directories for saving results
    save_path = os.path.join(SAVE_BASE_PATH, target_model_name)
    for sub in ['ci', 'metrics']:
        os.makedirs(os.path.join(save_path, sub), exist_ok=True)

    # Get list of CSV files
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory not found: {DATA_DIR}")
        return

    file_list = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    
    for file_name in file_list:
        print(f"\n📂 Processing Dataset: {file_name}")
        
        # 1. Load Data
        # NOTE: Adjust split logic index [0] based on your actual filename format
        label_base = file_name.split("_")[0] 
        
        try:
            # Load Discovery Data
            df_disc = pd.read_csv(os.path.join(DATA_DIR, file_name), header=None)
            df_disc_lbl = pd.read_csv(os.path.join(LABEL_DIR, f"{label_base}_discorvery_label.csv"), header=None)
            X_disc, y_disc = df_disc.values, df_disc_lbl.values.flatten()
            
            # Load Validation Data
            # Assumes validation filenames match the discovery prefix
            df_val = pd.read_csv(os.path.join(VAL_DIR, f"{label_base}_valdataset.csv"), header=None)
            df_val_lbl = pd.read_csv(os.path.join(LABEL_DIR, f"{label_base}_validation_label.csv"), header=None)
            X_val, y_val = df_val.values, df_val_lbl.values.flatten()
            
        except FileNotFoundError as e:
            print(f"❌ Error loading files for {file_name}: {e}")
            continue

        print(f"   - Discovery set shape: {X_disc.shape}")
        print(f"   - Validation set shape: {X_val.shape}")

        # 2. Initialize Model & Params
        base_model, params = get_model_config(target_model_name, y_disc)
        
        all_metrics = []

        # 3. Iterate over Hyperparameters
        for param in params:
            # Create a string representation of params for filenames
            param_str = "__".join([f"{k}-{v}" for k, v in param.items()])
            
            # Set parameters
            base_model.set_params(**param)
            
            # --- [Strategy] Stratified Shuffle Split (5 splits) ---
            sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=RNG_SEED)
            
            tprs_disc = []
            aucs_disc = []
            stats_disc = {'acc': [], 'pre': [], 'rec': [], 'f1': []}

            # ① Internal Validation (Discovery: 5-Fold Stratified Shuffle Split)
            for train_idx, test_idx in sss.split(X_disc, y_disc):
                X_train, y_train = X_disc[train_idx], y_disc[train_idx]
                X_test, y_test = X_disc[test_idx], y_disc[test_idx]

                base_model.fit(X_train, y_train)
                
                # Predictions
                y_prob = base_model.predict_proba(X_test)[:, 1]
                y_pred = base_model.predict(X_test)

                # Collect Metrics
                aucs_disc.append(roc_auc_score(y_test, y_prob))
                stats_disc['acc'].append(accuracy_score(y_test, y_pred))
                stats_disc['pre'].append(precision_score(y_test, y_pred, zero_division=0))
                stats_disc['rec'].append(recall_score(y_test, y_pred, zero_division=0))
                stats_disc['f1'].append(f1_score(y_test, y_pred, zero_division=0))

                # ROC Interpolation
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                interp_tpr = np.interp(MEAN_FPR, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs_disc.append(interp_tpr)

            # Calculate Discovery CI (Bootstrap on Fold TPRs)
            mean_tpr_disc = np.mean(tprs_disc, axis=0)
            mean_tpr_disc[-1] = 1.0
            
            boot_tprs = []
            for _ in range(N_BOOTSTRAPS):
                indices = RNG.choice(len(tprs_disc), len(tprs_disc), replace=True)
                sample_tprs = np.array(tprs_disc)[indices]
                boot_tprs.append(np.mean(sample_tprs, axis=0))
            
            boot_tprs = np.array(boot_tprs)
            ci_lower_disc = np.percentile(boot_tprs, 2.5, axis=0)
            ci_upper_disc = np.percentile(boot_tprs, 97.5, axis=0)

            # Save Discovery CI
            pd.DataFrame({
                'fpr': MEAN_FPR, 'tpr': mean_tpr_disc, 
                'lower_ci': ci_lower_disc, 'upper_ci': ci_upper_disc
            }).to_csv(os.path.join(save_path, f"ci/discovery_{file_name}_{param_str}.csv"), index=False)


            # ② External Validation (Train on Full Discovery -> Test on Validation)
            base_model.fit(X_disc, y_disc)
            y_val_prob = base_model.predict_proba(X_val)[:, 1]
            y_val_pred = base_model.predict(X_val)

            # Validation Metrics
            val_metrics = {
                'auc': roc_auc_score(y_val, y_val_prob),
                'acc': accuracy_score(y_val, y_val_pred),
                'pre': precision_score(y_val, y_val_pred, zero_division=0),
                'rec': recall_score(y_val, y_val_pred, zero_division=0),
                'f1': f1_score(y_val, y_val_pred, zero_division=0)
            }

            # Validation ROC & CI (Bootstrap on Samples)
            fpr_val, tpr_val, _ = roc_curve(y_val, y_val_prob)
            interp_tpr_val = np.interp(MEAN_FPR, fpr_val, tpr_val)
            interp_tpr_val[0] = 0.0

            boot_tprs_val = []
            for _ in range(N_BOOTSTRAPS):
                # Sample-level bootstrapping
                indices = RNG.choice(len(y_val), len(y_val), replace=True)
                if len(np.unique(y_val[indices])) < 2: continue # Skip if only one class selected
                
                y_sample = y_val[indices]
                prob_sample = y_val_prob[indices]
                
                fpr_i, tpr_i, _ = roc_curve(y_sample, prob_sample)
                interp_i = np.interp(MEAN_FPR, fpr_i, tpr_i)
                interp_i[0] = 0.0
                boot_tprs_val.append(interp_i)

            if boot_tprs_val:
                boot_tprs_val = np.array(boot_tprs_val)
                ci_lower_val = np.percentile(boot_tprs_val, 2.5, axis=0)
                ci_upper_val = np.percentile(boot_tprs_val, 97.5, axis=0)
            else:
                ci_lower_val = ci_upper_val = interp_tpr_val # Fallback

            # Save Validation CI
            pd.DataFrame({
                'fpr': MEAN_FPR, 'tpr': interp_tpr_val,
                'lower_ci': ci_lower_val, 'upper_ci': ci_upper_val
            }).to_csv(os.path.join(save_path, f"ci/validation_{file_name}_{param_str}.csv"), index=False)

            # ③ Compile Metrics
            all_metrics.append({
                'param': param_str,
                'disc_auc_mean': np.mean(aucs_disc), 'disc_auc_std': np.std(aucs_disc),
                'disc_acc_mean': np.mean(stats_disc['acc']), 'disc_acc_std': np.std(stats_disc['acc']),
                'disc_f1_mean': np.mean(stats_disc['f1']), 'disc_f1_std': np.std(stats_disc['f1']),
                'val_auc': val_metrics['auc'],
                'val_acc': val_metrics['acc'],
                'val_f1': val_metrics['f1']
            })

        # Save all metrics for this dataset
        pd.DataFrame(all_metrics).to_csv(os.path.join(save_path, f"metrics/{file_name}_all_metrics.csv"), index=False)
        print(f"   ✅ Saved metrics for {file_name}")

# ==========================================
# Execution Entry Point
# ==========================================
if __name__ == "__main__":
    # Select models to run ('svm', 'rf', 'knn', 'xgboost')
    target_models = ['svm', 'rf', 'knn', 'xgboost'] 
    
    for model in target_models:
        run_analysis(model)