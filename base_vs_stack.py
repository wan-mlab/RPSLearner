import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
from joblib import Parallel, delayed

from model import get_knn_model, get_rf_model, get_extra_trees_model, get_lightGBM_model, get_catboost_model, get_xgb_model, get_svm_model
from model import SimpleNNClassifier, RandomProjectionReducer, MetaStacker

def data_normalization(X):
    """
    This function normalizes the data using z-score normalization.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


def evaluate_models_predict(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs > threshold).astype(int)

    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'F1': f1_score(y_true, y_pred, average='weighted'),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_probs, average='weighted')
    }
    return metrics

def process_fold(
    fold_id,
    train_idx,
    test_idx,
    X,
    y,
    base_models,
    transformers,
    meta_model_choice,
    hidden_layers,
    inner_cv,
    passthrough
):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Apply random projection transforms, if available
    if transformers is not None:
        X_train_list = []
        X_test_list = []
        for transformer in transformers:
            X_train_list.append(transformer.fit_transform(X_train))
            X_test_list.append(transformer.transform(X_test))
        X_train_transformed = np.concatenate(X_train_list, axis=1)
        X_test_transformed = np.concatenate(X_test_list, axis=1)
    else:
        X_train_transformed = X_train
        X_test_transformed = X_test

    input_dim = X_train_transformed.shape[1]

    # Add a neural network as a base model
    local_base_models = base_models + [
        ("nn", SimpleNNClassifier(
            input_dim=input_dim,
            output_dim=1,
            epochs=100,
            batch_size=64,
            learning_rate=1e-4
        ))
    ]

    # Collect per-fold metrics here
    fold_predictions = {}

    # 1) Collect predictions from each base model
    for model_name, estimator in local_base_models:
        estimator.fit(X_train_transformed, y_train)
        y_probs_base = estimator.predict_proba(X_test_transformed)[:, 1]
        fold_predictions[model_name] = y_probs_base

    def get_meta_model(input_dim_for_meta):
        if meta_model_choice == "nn":
            # If passthrough=True, the meta input is (base outputs + raw features).
            meta_input_dim = len(local_base_models) + input_dim_for_meta if passthrough else len(local_base_models)
            return MetaStacker(
                input_dim=meta_input_dim,
                output_dim=1,
                epochs=1000,
                batch_size=64,
                learning_rate=1e-4,
                hidden_layers=hidden_layers
            )
        elif meta_model_choice == "rf":
            return get_rf_model(criterion="gini")
        elif meta_model_choice == "svm":
            return get_svm_model(kernel="rbf")
        elif meta_model_choice == "lr":
            return LogisticRegression()
        else:
            raise ValueError(f"Unknown meta model choice: {meta_model_choice}")

    meta_model = get_meta_model(input_dim)

    from sklearn.ensemble import StackingClassifier
    stacking_clf = StackingClassifier(
        estimators=local_base_models,
        final_estimator=meta_model,
        passthrough=passthrough,
        cv=inner_cv,
        n_jobs=5  # parallel inside stacking (base models in cross-val)
    )

    stacking_clf.fit(X_train_transformed, y_train)
    y_probs_meta = stacking_clf.predict_proba(X_test_transformed)[:, 1]

    fold_predictions[f"meta_{meta_model_choice}"] = y_probs_meta
    fold_predictions["test_indices"] = test_idx

    return fold_predictions


def compare_base_models_vs_meta_parallel(
    X,
    y,
    n_splits=5,
    random_state=42,
    use_RP=True,
    k=20,
    n_components=200,
    seedn=42,
    SSSE=False,
    meta_model_choice="nn",
    hidden_layers=4,
    inner_cv=5,
    passthrough=True,
    n_jobs_outer=-1  # number of CPU cores for parallelizing folds
):
    """
    Compare base models vs. meta-model with optional random projections,
    using joblib to parallelize the *outer* folds.
    """
    X = np.log1p(X)
    label_mapping = {"LUAD": 0, "LUSC": 1}
    y = y.map(label_mapping)
    y = y.values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Predefine the random projection transformers (if using RP)
    if use_RP:
        # transformers = [
        #     RandomProjectionReducer(
        #         reduced_dim=n_components,
        #         seedn=s*2,
        #         SSSE=SSSE
        #     )
        #     for s in range(seedn, seedn + k)
        # ]
        transformers = [
            GaussianRandomProjection(n_components=n_components, random_state=s)
            for s in range(seedn, seedn + k)
        ]
    else:
        transformers = None

    # Define your base models once
    base_models = [
        ("knn_distance", get_knn_model(weights="distance")),
        ("knn_uniform", get_knn_model(weights="uniform")),
        ("rf_gini", get_rf_model(criterion="gini")),
        ("rf_entropy", get_rf_model(criterion="entropy")),
        ("extra_trees_gini", get_extra_trees_model(criteria="gini")),
        ("extra_trees_entropy", get_extra_trees_model(criteria="entropy")),
        ("lightgbm", get_lightGBM_model()),
        ("catboost", get_catboost_model()),
        ("xgboost", get_xgb_model()),
    ]

    # Prepare parallel jobs
    # Each job processes one fold via the process_fold function above.
    jobs = (
        delayed(process_fold)(
            fold_id,
            train_idx,
            test_idx,
            X,
            y,
            base_models,
            transformers,
            meta_model_choice,
            hidden_layers,
            inner_cv,
            passthrough
        )
        for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, y))
    )

    # Run all fold jobs in parallel
    folds_results_list = Parallel(n_jobs=n_jobs_outer, verbose=10)(jobs)

    # Flatten the results into a single list of dicts
    model_names = [name for name, _ in base_models] + ["nn", f"meta_{meta_model_choice}"]
    predictions_dict = {model_name: np.zeros(len(y)) for model_name in model_names}

    # Populate the predictions_dict with predictions from each fold
    for fold_result in folds_results_list:
        test_idx = fold_result["test_indices"]
        for model_name in model_names:
            predictions_dict[model_name][test_idx] = fold_result[model_name]

    overall_metrics = {}
    # folds_results_list is a list of lists (one sub-list per fold).
    for model_name in model_names:
        y_probs = predictions_dict[model_name]
        y_pred = (y_probs > 0.5).astype(int)  # You can choose a different threshold if needed

        metrics = {
            'Accuracy': accuracy_score(y, y_pred),
            'Precision': precision_score(y, y_pred, average='weighted'),
            'Recall': recall_score(y, y_pred, average='weighted'),
            'F1': f1_score(y, y_pred, average='weighted'),
            'MCC': matthews_corrcoef(y, y_pred),
            'AUC': roc_auc_score(y, y_probs, average='weighted')
        }

        overall_metrics[model_name] = metrics

    # Convert to DataFrame
    results_df = pd.DataFrame(overall_metrics).T
    return results_df


def process_fold_combination(
    fold_id,
    train_idx,
    test_idx,
    X,
    y,
    base_models,
    transformers,
    meta_model_choice,
    hidden_layers,
    inner_cv,
    passthrough
):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Apply random projection transforms, if available
    if transformers is not None:
        X_train_list = []
        X_test_list = []
        for transformer in transformers:
            X_train_list.append(transformer.fit_transform(X_train))
            X_test_list.append(transformer.transform(X_test))
        X_train_transformed = np.concatenate(X_train_list, axis=1)
        X_test_transformed = np.concatenate(X_test_list, axis=1)
    else:
        X_train_transformed = X_train
        X_test_transformed = X_test

    input_dim = X_train_transformed.shape[1]

    # Collect per-fold metrics here
    fold_predictions = {}

    def get_meta_model(input_dim_for_meta):
        if meta_model_choice == "nn":
            # If passthrough=True, the meta input is (base outputs + raw features).
            meta_input_dim = len(base_models) + input_dim_for_meta if passthrough else len(base_models)
            return MetaStacker(
                input_dim=meta_input_dim,
                output_dim=1,
                epochs=1000,
                batch_size=64,
                learning_rate=1e-4,
                hidden_layers=hidden_layers
            )
        elif meta_model_choice == "rf":
            return get_rf_model(criterion="gini")
        elif meta_model_choice == "svm":
            return get_svm_model(kernel="rbf")
        elif meta_model_choice == "lr":
            return LogisticRegression()
        else:
            raise ValueError(f"Unknown meta model choice: {meta_model_choice}")

    meta_model = get_meta_model(input_dim)

    from sklearn.ensemble import StackingClassifier
    stacking_clf = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        passthrough=passthrough,
        cv=inner_cv,
        n_jobs=5  # parallel inside stacking (base models in cross-val)
    )

    stacking_clf.fit(X_train_transformed, y_train)
    y_probs_meta = stacking_clf.predict_proba(X_test_transformed)[:, 1]

    combination_size = len(base_models)
    fold_predictions["model_name"] = f"meta_{meta_model_choice}_{combination_size}"
    fold_predictions[f"meta_{meta_model_choice}_{combination_size}"] = y_probs_meta
    fold_predictions["test_indices"] = test_idx

    return fold_predictions


def composite_compare_base_models(
    X,
    y,
    n_splits=5,
    random_state=42,
    use_RP=True,
    k=20,
    n_components=200,
    seedn=42,
    SSSE=False,
    meta_model_choice="nn",
    hidden_layers=4,
    inner_cv=5,
    passthrough=True,
    n_jobs_outer=-1  # number of CPU cores for parallelizing folds
):
    # Compare base models combination
    X = np.log1p(X)
    label_mapping = {"LUAD": 0, "LUSC": 1}
    y = y.map(label_mapping)
    y = y.values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Predefine the random projection transformers (if using RP)
    if use_RP:
        transformers = [
            GaussianRandomProjection(n_components=n_components, random_state=s)
            for s in range(seedn, seedn + k)
        ]
    else:
        transformers = None

    # Define your base models once
    base_models = [
        ("knn_distance", get_knn_model(weights="distance")),
        ("knn_uniform", get_knn_model(weights="uniform")),
        ("rf_gini", get_rf_model(criterion="gini")),
        ("extra_trees_gini", get_extra_trees_model(criteria="gini")),
        ("extra_trees_entropy", get_extra_trees_model(criteria="entropy")),
        ("lightgbm", get_lightGBM_model()),
        ("xgboost", get_xgb_model()),
        ("rf_entropy", get_rf_model(criterion="entropy")),
        ("catboost", get_catboost_model()),
        ("nn", SimpleNNClassifier(
            input_dim=X.shape[1],
            output_dim=1,
            epochs=100,
            batch_size=64,
            learning_rate=1e-4
        )),
    ]

    # reverse the order of base models
    base_models.reverse()
    # create base models combinations, each time, add a new model to the model list
    base_models_combinations = []
    for i in range(1, len(base_models)):
        top_n_models = base_models[:i]
        base_models_combinations.append(top_n_models)
    
    # Prepare parallel jobs
    # Each job processes one fold via the process_fold function above.
    jobs = (
        delayed(process_fold_combination)(
            fold_id,
            train_idx,
            test_idx,
            X,
            y,
            base_models_combination,
            transformers,
            meta_model_choice,
            hidden_layers,
            inner_cv,
            passthrough
        )
        for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, y))
        for base_models_combination in base_models_combinations
    )

    # Run all fold jobs in parallel
    folds_results_list = Parallel(n_jobs=n_jobs_outer)(jobs)

    # Flatten the results into a single list of dicts
    # model_names = [name for name, _ in base_models] + ["nn", f"meta_{meta_model_choice}"]
    base_combination_names = [
        f"meta_{meta_model_choice}_{len(base_models_combination)}" for base_models_combination in base_models_combinations
        ]
    predictions_dict = {combination_name: np.zeros(len(y)) for combination_name in base_combination_names}

    # Populate the predictions_dict with predictions from each fold
    for fold_result in folds_results_list:
        test_idx = fold_result["test_indices"]
        model_name = fold_result["model_name"]
        predictions_dict[model_name][test_idx] = fold_result[model_name]

    overall_metrics = {}
    # folds_results_list is a list of lists (one sub-list per fold).
    for combination_name in base_combination_names:
        y_probs = predictions_dict[combination_name]
        y_pred = (y_probs > 0.5).astype(int)  # You can choose a different threshold if needed

        metrics = {
            'Accuracy': accuracy_score(y, y_pred),
            'Precision': precision_score(y, y_pred, average='weighted'),
            'Recall': recall_score(y, y_pred, average='weighted'),
            'F1': f1_score(y, y_pred, average='weighted'),
            'MCC': matthews_corrcoef(y, y_pred),
            'AUC': roc_auc_score(y, y_probs, average='weighted')
        }

        overall_metrics[combination_name] = metrics

    # Convert to DataFrame
    results_df = pd.DataFrame(overall_metrics).T
    return results_df