import os
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from collections import Counter
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score

# ==========================================
# Configuration & Paths
# ==========================================
BASE_DIR = os.getcwd()  # 현재 작업 디렉토리
DATA_DIR = os.path.join(BASE_DIR, "data")       # 데이터 폴더 (input)
RESULT_DIR = os.path.join(BASE_DIR, "results")  # 결과 폴더 (output)

# 분석할 데이터셋 리스트 (파일명에 맞게 수정)
DATASETS = [
    "g1_KO", "g1_species", "g1_speciesKO",
    "g2_KO", "g2_species", "g2_speciesKO"
]

# 결과 폴더 생성
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

# ==========================================
# Helper Functions
# ==========================================
def load_data(dataset_name):
    """
    Loads features (X), labels (y), and feature names.
    Assumes CSV format without headers for X and y, and a separate file for feature names.
    """
    try:
        # 경로 설정 (사용자 환경에 맞게 파일명 패턴 수정 필요)
        x_path = os.path.join(DATA_DIR, f"{dataset_name}_MLdataset.csv")
        y_path = os.path.join(DATA_DIR, f"{dataset_name}_label.csv")
        feat_path = os.path.join(DATA_DIR, f"{dataset_name}_features.csv") # feature 이름 파일

        X = pd.read_csv(x_path, header=None).values
        y = pd.read_csv(y_path, header=None).values.flatten()
        
        # Feature 이름 로드 (없으면 임의 생성)
        if os.path.exists(feat_path):
            feature_names = pd.read_csv(feat_path, header=None)[0].tolist()
        else:
            feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
            
        return X, y, feature_names
    except FileNotFoundError as e:
        print(f"Error loading {dataset_name}: {e}")
        return None, None, None

def get_scale_pos_weight(y):
    """Calculates scale_pos_weight for imbalanced classes."""
    counter = Counter(y)
    return counter[0] / counter[1]

def tune_hyperparameters(X, y, scale_weight):
    """Performs Grid Search to find optimal XGBoost parameters."""
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric="logloss",
        scale_pos_weight=scale_weight,
        n_jobs=-1
    )

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'colsample_bytree': [0.5, 0.8, 1.0]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(xgb_model, param_grid, scoring='roc_auc', cv=cv, n_jobs=-1, verbose=0)
    grid.fit(X, y)
    
    return grid.best_params_

# ==========================================
# Main Analysis Loop
# ==========================================
def main():
    for data_name in DATASETS:
        print(f"\nProcessing: {data_name}...")
        
        # 1. Load Data
        X, y, feature_names = load_data(data_name)
        if X is None: continue

        # 2. Handle Class Imbalance
        weight = get_scale_pos_weight(y)
        
        # 3. Hyperparameter Tuning
        print("  > Tuning hyperparameters...")
        best_params = tune_hyperparameters(X, y, weight)
        print(f"  > Best Params: {best_params}")

        # 4. Stratified 10-Fold CV with SHAP
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        fold_metrics = []
        all_shap_values = []
        all_X_test = []  # For SHAP summary
        
        print("  > Running 10-Fold CV & SHAP Analysis...")
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Train Model with Best Params
            model = xgb.XGBClassifier(
                **best_params,
                objective='binary:logistic',
                use_label_encoder=False,
                eval_metric="logloss",
                scale_pos_weight=weight,
                n_jobs=-1
            )
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            # Metrics
            fold_metrics.append({
                "Fold": fold + 1,
                "AUC": roc_auc_score(y_test, y_prob),
                "Accuracy": accuracy_score(y_test, y_pred),
                "F1_Score": f1_score(y_test, y_pred)
            })

            # SHAP Calculation
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_test)
            
            # Collect SHAP values and corresponding features
            if isinstance(shap_vals, list): # For some XGB versions
                shap_vals = shap_vals[1]
            
            # Create DataFrame for this fold
            fold_shap_df = pd.DataFrame(shap_vals, columns=feature_names)
            all_shap_values.append(fold_shap_df)

        # ==========================================
        # 5. Save Results
        # ==========================================
        
        # 5-1. Save Metrics
        metrics_df = pd.DataFrame(fold_metrics)
        metrics_summary = pd.DataFrame({
            "Metric": ["AUC", "Accuracy", "F1_Score"],
            "Mean": [metrics_df["AUC"].mean(), metrics_df["Accuracy"].mean(), metrics_df["F1_Score"].mean()],
            "Std": [metrics_df["AUC"].std(), metrics_df["Accuracy"].std(), metrics_df["F1_Score"].std()]
        })
        
        save_path_metrics = os.path.join(RESULT_DIR, f"{data_name}_metrics.csv")
        metrics_summary.to_csv(save_path_metrics, index=False)
        print(f"  > Metrics saved to {save_path_metrics}")

        # 5-2. Save Consolidated SHAP Values
        # (Concatenate all folds to represent the whole dataset)
        total_shap_df = pd.concat(all_shap_values, axis=0).reset_index(drop=True)
        
        # Calculate Mean Absolute SHAP (Global Importance)
        feature_importance = pd.DataFrame({
            "Feature": feature_names,
            "Mean_Abs_SHAP": total_shap_df.abs().mean().values
        }).sort_values(by="Mean_Abs_SHAP", ascending=False)

        save_path_shap = os.path.join(RESULT_DIR, f"{data_name}_SHAP_importance.csv")
        feature_importance.to_csv(save_path_shap, index=False)
        
        # Save Raw SHAP values (Optional, for reproducibility)
        save_path_raw_shap = os.path.join(RESULT_DIR, f"{data_name}_SHAP_raw.csv")
        total_shap_df.to_csv(save_path_raw_shap, index=False)
        
        print(f"  > SHAP data saved to {save_path_shap}")

if __name__ == "__main__":
    main()