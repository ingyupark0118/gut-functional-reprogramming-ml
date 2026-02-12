import os
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 서버 환경 호환 (창 띄우기 방지)
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    roc_auc_score, accuracy_score, recall_score, precision_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve, average_precision_score, auc
)

# ==========================================
# 1. Configuration (설정)
# ==========================================
class Config:
    # 경로 설정 (현재 스크립트 위치 기준 상대 경로 권장)
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, 'ML_dataset', 'disc')  # feature 파일 위치
    LABEL_DIR = os.path.join(BASE_DIR, 'ML_dataset')         # label 파일 위치
    SAVE_DIR = os.path.join(BASE_DIR, 'results', 'roc')      # 결과 저장 위치
    
    # 데이터 관련
    LABEL_FILENAME = 'discovery_label.csv' # 라벨 파일명이 고정이라면 지정
    GROUP_NAMES = ['NC_non', 'DM_non', 'NC_cirr', 'DM_cirr']
    CLASS_COLORS = {0: "#317EC2", 1: "#5EB35E", 2: "#E89C31", 3: "#C03830"}
    
    # 모델 관련
    RANDOM_STATE = 42
    N_JOBS = 20 # CPU 코어 수에 맞게 조절
    CV_FOLDS = 5 # 논문 표준 (보통 5 or 10)

    # 하이퍼파라미터 그리드
    RF_PARAMS = [{
        'n_estimators': [100, 300, 500],
        'max_features': ['sqrt', 'log2'],
        'min_samples_leaf': [3, 5],
        'max_depth': [3, 5, None],
        'bootstrap': [True],
        'criterion': ['gini', 'entropy']
    }]

# 전역 폰트 설정
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.5

# ==========================================
# 2. Helper Functions (기능 함수)
# ==========================================

def make_dirs(path):
    """결과 저장용 폴더 트리 생성"""
    for sub in ['ci', 'metrics', 'confusion', 'roc']:
        os.makedirs(os.path.join(path, sub), exist_ok=True)

def load_dataset(feature_file):
    """데이터 로드 및 매칭 (Feature + Label)"""
    try:
        # Feature Load
        f_path = os.path.join(Config.DATA_DIR, feature_file)
        df = pd.read_csv(f_path, sep=',', encoding='utf-8-sig', header=None)
        X = df.values

        # Label Load (라벨 파일이 하나로 고정된 경우)
        l_path = os.path.join(Config.LABEL_DIR, Config.LABEL_FILENAME)
        
        # 만약 파일마다 라벨이 다르다면 아래 주석 해제하여 사용:
        # label_base = feature_file.split("_")[0]
        # l_path = os.path.join(Config.LABEL_DIR, f"{label_base}_label.csv")

        with open(l_path, 'r', encoding='utf-8') as f:
            y = np.array([int(row[0]) for row in csv.reader(f)])
            
        return X, y
    except Exception as e:
        print(f"[Error] Failed to load {feature_file}: {e}")
        return None, None

def get_best_model(X, y):
    """GridSearchCV를 통한 최적 모델 탐색"""
    rf = RandomForestClassifier(class_weight='balanced', random_state=Config.RANDOM_STATE, n_jobs=Config.N_JOBS)
    grid = GridSearchCV(
        estimator=rf,
        param_grid=Config.RF_PARAMS,
        scoring='roc_auc_ovr',
        cv=StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE),
        n_jobs=-1,
        verbose=0
    )
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_

# ==========================================
# 3. Main Analysis Logic
# ==========================================

def run_analysis():
    make_dirs(Config.SAVE_DIR)
    
    file_list = [f for f in os.listdir(Config.DATA_DIR) if f.endswith('.csv')]
    all_best_models_summary = []

    for file_name in file_list:
        print(f"\nProcessing: {file_name}...")
        
        # 1. 데이터 로드
        X, y = load_dataset(file_name)
        if X is None: continue
        
        classes = np.unique(y)
        n_classes = len(classes)
        
        # 2. 모델 학습 (Best Param 찾기)
        clf, best_params = get_best_model(X, y)
        param_str = f"est{best_params['n_estimators']}_dp{best_params['max_depth']}" # 파일명용 짧은 파라미터
        print(f"  > Best Params: {best_params}")

        # 3. Cross-Validation & Metric Collection
        cv = StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)
        
        # 결과 저장용 리스트
        results = {
            'auc': [], 'acc': [], 'f1': [],
            'fold_data': [], # (y_test_bin, y_proba) 저장용
            'conf_matrix': []
        }

        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)
            
            # Label Binarize (for ROC/PR)
            y_test_bin = label_binarize(y_test, classes=classes)
            if n_classes == 2 and y_test_bin.shape[1] == 1:
                y_test_bin = np.hstack((1 - y_test_bin, y_test_bin))

            # Store per-fold data
            results['fold_data'].append((y_test_bin, y_proba))
            results['auc'].append(roc_auc_score(y_test, y_proba, multi_class='ovr'))
            results['acc'].append(accuracy_score(y_test, y_pred))
            results['f1'].append(f1_score(y_test, y_pred, average='macro'))
            results['conf_matrix'].append(confusion_matrix(y_test, y_pred, labels=classes))

        # 4. Save Base Metrics
        # Confusion Matrix
        sum_cm = np.sum(results['conf_matrix'], axis=0)
        pd.DataFrame(sum_cm, index=classes, columns=classes).to_csv(
            os.path.join(Config.SAVE_DIR, "confusion", f"cm_{file_name}_{param_str}.csv"))

        # Summary Metrics
        metrics_df = pd.DataFrame([{
            'Dataset': file_name, 'AUC_mean': np.mean(results['auc']), 'AUC_std': np.std(results['auc']),
            'ACC_mean': np.mean(results['acc']), 'F1_mean': np.mean(results['f1'])
        }])
        metrics_df.to_csv(os.path.join(Config.SAVE_DIR, "metrics", f"metrics_{file_name}_{param_str}.csv"), index=False)

        # 5. Plotting (ROC & PR Curve) - 핵심 수정 부분
        fig1, ax1 = plt.subplots(figsize=(8, 8)) # ROC
        fig2, ax2 = plt.subplots(figsize=(8, 8)) # PR

        mean_fpr = np.linspace(0, 1, 100)
        recall_interp = np.linspace(0, 1, 100) # PR Curve용 X축 (0~1)
        z_score = norm.ppf(0.975) # 95% CI
        
        mean_auc_dict = {}

        # Class별 반복 (중요: 리스트 초기화 위치 주의)
        for i, label in enumerate(classes):
            tprs = []
            precs = []
            aucs = []
            aps = []

            # 각 Fold의 결과를 가져와서 처리
            for y_bin_fold, proba_fold in results['fold_data']:
                # ROC
                fpr, tpr, _ = roc_curve(y_bin_fold[:, i], proba_fold[:, i])
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                aucs.append(roc_auc_score(y_bin_fold[:, i], proba_fold[:, i]))

                # PR
                prec, rec, _ = precision_recall_curve(y_bin_fold[:, i], proba_fold[:, i])
                # PR Interpolation (Recall 축 기준, 역순 주의)
                precs.append(np.interp(recall_interp, rec[::-1], prec[::-1]))
                aps.append(average_precision_score(y_bin_fold[:, i], proba_fold[:, i]))

            # Statistics Calculation
            # ROC
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = np.mean(aucs)
            mean_auc_dict[label] = mean_auc
            std_tpr = np.std(tprs, axis=0)
            ci_roc = z_score * (std_tpr / np.sqrt(len(tprs)))

            # PR
            mean_prec = np.mean(precs, axis=0)
            mean_ap = np.mean(aps)
            std_prec = np.std(precs, axis=0)
            ci_pr = z_score * (std_prec / np.sqrt(len(precs)))

            # Plotting ROC
            color = Config.CLASS_COLORS.get(label, 'black')
            ax1.plot(mean_fpr, mean_tpr, color=color, lw=3,
                     label=f'{Config.GROUP_NAMES[label]} (AUC={mean_auc:.2f})')
            ax1.fill_between(mean_fpr, np.maximum(mean_tpr - ci_roc, 0), np.minimum(mean_tpr + ci_roc, 1),
                             color=color, alpha=0.1)

            # Plotting PR (수정: fill_between x축을 recall_interp로 변경)
            ax2.plot(recall_interp, mean_prec, color=color, lw=3,
                     label=f'{Config.GROUP_NAMES[label]} (AP={mean_ap:.2f})')
            ax2.fill_between(recall_interp, np.maximum(mean_prec - ci_pr, 0), np.minimum(mean_prec + ci_pr, 1),
                             color=color, alpha=0.1)

            # CI Data Save (Optional)
            pd.DataFrame({
                'fpr': mean_fpr, 'tpr': mean_tpr, 'lower': mean_tpr - ci_roc, 'upper': mean_tpr + ci_roc
            }).to_csv(os.path.join(Config.SAVE_DIR, "ci", f"ci_roc_{file_name}_class{label}.csv"), index=False)

        # Graph Styling
        for ax, title in zip([ax1, ax2], ['ROC Curve', 'Precision-Recall Curve']):
            ax.plot([0, 1], [0, 1] if title == 'ROC Curve' else [1, 0], 'k--', alpha=0.5) # Diagonal
            ax.set_xlabel('False Positive Rate' if title == 'ROC Curve' else 'Recall')
            ax.set_ylabel('True Positive Rate' if title == 'ROC Curve' else 'Precision')
            ax.set_title(title)
            ax.legend(loc="lower right" if title == 'ROC Curve' else "lower left", frameon=False, fontsize=12)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        fig1.tight_layout()
        fig1.savefig(os.path.join(Config.SAVE_DIR, "roc", f"ROC_{file_name}_{param_str}.png"), dpi=300)
        fig2.tight_layout()
        fig2.savefig(os.path.join(Config.SAVE_DIR, "roc", f"PRC_{file_name}_{param_str}.png"), dpi=300)
        plt.close('all')

        # Best Model Summary Add
        all_best_models_summary.append({
            'Dataset': file_name, 'Params': best_params, **mean_auc_dict
        })

    # Final Summary Save
    pd.DataFrame(all_best_models_summary).to_csv(os.path.join(Config.SAVE_DIR, "final_summary.csv"), index=False)

if __name__ == "__main__":
    run_analysis()