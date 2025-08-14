import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from joblib import Parallel, delayed

def ranM(scdata, p, seedn=42):
    # Random projection matrix generator
    m = scdata.shape[0]  # number of features
    s = np.sqrt(m)

    np.random.seed(int(seedn))
    x0 = np.random.choice([np.sqrt(s), 0, -np.sqrt(s)], size=m * p, replace=True, p=[1/(2*s), 1 - 1/s, 1/(2*s)])

    x = csr_matrix(x0.reshape(m, p), dtype=np.float32)
    return x

# def train_and_evaluate(tpm_data, labels, cv_seed=42):
#     # log transform
#     tpm_data = np.log1p(tpm_data)

#     # label mapping
#     label_mapping = {"LUAD": 0, "LUSC": 1}
#     labels = labels.map(label_mapping).values

#     all_probabilities = []
#     all_true_labels = []
    
#     # Cross-validation setup
#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cv_seed)
#     P = [600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500]

#     for train_index, test_index in skf.split(tpm_data, labels):
#         X_train, X_test = tpm_data.iloc[train_index], tpm_data.iloc[test_index]
#         y_train, y_test = labels[train_index], labels[test_index]

#         for p in P:
#             # Apply random projection
#             oritpm = np.array(X_train.T)
#             rM = ranM(X_train.T, p, 1)
#             X_train_proj = (1 / np.sqrt(p)) * np.transpose(rM) @ oritpm
#             X_train_proj = np.transpose(X_train_proj)

#             oritpm_test = np.array(X_test.T)
#             X_test_proj = (1 / np.sqrt(p)) * np.transpose(rM) @ oritpm_test
#             X_test_proj = np.transpose(X_test_proj)

#             # Standardize data
#             scaler = StandardScaler()
#             X_train_scaled = scaler.fit_transform(X_train_proj)
#             X_test_scaled = scaler.transform(X_test_proj)

#             # Train SVM
#             model = SVC(probability=True, kernel='linear', random_state=42)
#             model.fit(X_train_scaled, y_train)

#             # Store probabilities and true labels
#             y_proba = model.predict_proba(X_test_scaled)[:, 1]
#             all_probabilities.extend(y_proba)
#             all_true_labels.extend(y_test)

#     # Compute overall metrics
#     all_probabilities = np.array(all_probabilities)
#     all_true_labels = np.array(all_true_labels)
    
#     acc = accuracy_score(all_true_labels, all_probabilities > 0.5)
#     f1 = f1_score(all_true_labels, all_probabilities > 0.5)
#     mcc = matthews_corrcoef(all_true_labels, all_probabilities > 0.5)
#     auc = roc_auc_score(all_true_labels, all_probabilities)

#     metrics = {'Accuracy': acc, 'F1': f1, 'MCC': mcc, 'AUC': auc}
#     metrics_df = pd.DataFrame([metrics])
#     probabilities_df = pd.DataFrame({'True_Label': all_true_labels, 'Probability': all_probabilities})

#     # metrics_df.to_csv('CV_Metrics.csv', index=False)
#     # probabilities_df.to_csv('Prediction_Probabilities.csv', index=False)

#     print(metrics_df)
#     # print(probabilities_df)
#     return metrics_df, probabilities_df



def process_fold(train_index, test_index, tpm_data, labels, P):
    final_prob = 0

    X_train, X_test = tpm_data.iloc[train_index], tpm_data.iloc[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    for p in P:
        print(f"Processing dimension: {p}")
        # Apply random projection
        oritpm = np.array(X_train.T)
        rM = ranM(X_train.T, p, seedn=1)
        X_train_proj = (1 / np.sqrt(p)) * np.transpose(rM) @ oritpm
        X_train_proj = np.transpose(X_train_proj)

        oritpm_test = np.array(X_test.T)
        X_test_proj = (1 / np.sqrt(p)) * np.transpose(rM) @ oritpm_test
        X_test_proj = np.transpose(X_test_proj)

        # Standardize data
        # scaler = StandardScaler()
        # X_train_scaled = scaler.fit_transform(X_train_proj)
        # X_test_scaled = scaler.transform(X_test_proj)

        # Train SVM
        model = SVC(probability=True, kernel='linear', random_state=42)
        model.fit(X_train_proj, y_train)

        # Store probabilities and true labels
        y_proba = model.predict_proba(X_test_proj)[:, 1]
        final_prob += y_proba

    fold_true_labels = y_test
    final_prob = final_prob / len(P)

    return final_prob, fold_true_labels

def train_and_evaluate(tpm_data, labels, dim = 100, cv_seed=42):
    # normalize data
    tpm_data = (tpm_data - tpm_data.mean()) / tpm_data.std()
    tpm_data = np.log1p(tpm_data)

    label_mapping = {"LUAD": 0, "LUSC": 1}
    labels = labels.map(label_mapping).values

    all_probabilities = []
    all_true_labels = []
    
    # Cross-validation setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cv_seed)
    # P = [600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500]
    P = [dim] * 10

    results = Parallel(n_jobs=5)(
        delayed(process_fold)(train_index, test_index, tpm_data, labels, P)
        for train_index, test_index in skf.split(tpm_data, labels)
    )

    for fold_probabilities, fold_true_labels in results:
        all_probabilities.extend(fold_probabilities)
        all_true_labels.extend(fold_true_labels)

    # Compute overall metrics
    all_probabilities = np.array(all_probabilities)
    all_true_labels = np.array(all_true_labels)
    
    acc = accuracy_score(all_true_labels, all_probabilities > 0.5)
    f1 = f1_score(all_true_labels, all_probabilities > 0.5)
    mcc = matthews_corrcoef(all_true_labels, all_probabilities > 0.5)
    auc = roc_auc_score(all_true_labels, all_probabilities)

    metrics = {'Accuracy': acc, 'F1': f1, 'MCC': mcc, 'AUC': auc}
    metrics_df = pd.DataFrame([metrics])
    probabilities_df = pd.DataFrame({'True_Label': all_true_labels, 'Probability': all_probabilities})

    # metrics_df.to_csv('CV_Metrics.csv', index=False)
    # probabilities_df.to_csv('Prediction_Probabilities.csv', index=False)

    print(metrics_df)
    # print(probabilities_df)
    return metrics_df, probabilities_df