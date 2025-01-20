import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
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

def process_fold(fold_id, train_index, test_index, X, y, **kwargs):
    print(f"Processing fold {fold_id+1}...")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Apply random projection
    use_RP = kwargs.get('use_RP', True)
    if use_RP:
        k = kwargs.get('k', 10)
        n_components = kwargs.get('n_components', 100)
        seedns = kwargs.get('seedn', 42)
        SSSE = kwargs.get('SSSE', False)
        transformers = [RandomProjectionReducer(
                reduced_dim=n_components, seedn=seedn*2, SSSE=SSSE) for seedn in range(seedns, seedns+k)
            ]
        X_train_transformed = [transformer.fit_transform(X_train) for transformer in transformers]
        X_train_transformed = np.concatenate(X_train_transformed, axis=1)
        X_test_transformed = [transformer.transform(X_test) for transformer in transformers]
        X_test_transformed = np.concatenate(X_test_transformed, axis=1)
    else:
        X_train_transformed, X_test_transformed = X_train, X_test

    input_dim = X_train_transformed.shape[1]

    # Redefine base models with updated input dimensions
    base_models = [
        ('knn_distance', get_knn_model(weights='distance')),
        ('knn_uniform', get_knn_model(weights='uniform')),
        ('rf_gini', get_rf_model(criterion='gini')),
        ('rf_entropy', get_rf_model(criterion='entropy')),
        ('extra_trees_gini', get_extra_trees_model(criteria='gini')),
        ('extra_trees_entropy', get_extra_trees_model(criteria='entropy')),
        ('lightgbm', get_lightGBM_model()),
        ('catboost', get_catboost_model()),
        ('xgboost', get_xgb_model()),
        ('nn', SimpleNNClassifier(input_dim=input_dim, output_dim=1, epochs=1000, batch_size=64, learning_rate=1e-4)),
    ]

    meta_model_choice = kwargs.get('meta_model', "nn")

    if meta_model_choice == "nn":
        meta_input_dim = len(base_models) + input_dim if kwargs.get('passthrough', True) else len(base_models)
        hidden_layers = kwargs.get('hidden_layers', 4)
        meta_model = MetaStacker(
                input_dim=meta_input_dim, output_dim=1, epochs=1000, 
                batch_size=64, learning_rate=1e-4, hidden_layers=hidden_layers
            )
    elif meta_model_choice == "rf":
        meta_model = get_rf_model(criterion='gini' if kwargs.get('criterion', 'gini') else 'entropy')
    elif meta_model_choice == "svm":
        meta_model = get_svm_model(kernel='rbf')
    elif meta_model_choice == "lr":
        # linear regression
        meta_model = LogisticRegression()
    else:
        raise ValueError(f"Unknown meta model choice: {meta_model_choice}")

    stacking_clf = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        passthrough=kwargs.get('passthrough', True),
        cv=kwargs.get('inner_cv', 5),
        n_jobs=5
    )

    stacking_clf.fit(X_train_transformed, y_train)
    y_probs = stacking_clf.predict_proba(X_test_transformed)[:, 1]  # Assuming binary classification
    return y_test, y_probs


def autogluon_stacking_framework(X, y, **kwargs):
    from autogluon.tabular import TabularPredictor
    if kwargs.get('log_transform', True):
        X = np.log1p(X)

    if kwargs.get('normalize', False):
        X = data_normalization(X)

    # Random Projection
    n_RP = kwargs.get('n_RP', 5)
    SSSE = kwargs.get('SSSE', True)
    reduced_dim = kwargs.get('reduced_dim', 50)
    concat = kwargs.get('concat', True)
    problem_type = kwargs.get('problem_type', 'binary')

    # generate random projections
    RP_transformers = [RandomProjectionReducer(
        reduced_dim=reduced_dim,
        seedn=i,
        SSSE=SSSE
    ) for i in range(n_RP)]

    # Apply reduction
    X_RPs = [RP.fit_transform(X) for RP in RP_transformers]

    # Prepare the data
    y_df = y.reset_index(drop=True)
    if isinstance(y_df, pd.Series):
        y_df = y_df.to_frame()
    if 'cancer_subtype' not in y_df.columns:
        y_df.columns = ['cancer_subtype']
    
    if concat:
        X_RPs = np.concatenate(X_RPs, axis=1)
        X_df = pd.DataFrame(X_RPs)
        data_df_RPs = pd.concat([X_df, y_df], axis=1)
    else:
        X_df_RPs = [pd.DataFrame(X_RP) for X_RP in X_RPs]
        data_df_RPs = [pd.concat([X_df_RP, y_df], axis=1) for X_df_RP in X_df_RPs]

    # Train and evaluate models
    model_tmp_directory = '/work/wanlab/xinchaowu/Lung_Cancer_subtypes/tmp/ag_models'
    model_tmp_directory = kwargs.get('model_tmp_directory', model_tmp_directory)
    time_limit = kwargs.get('time_limit', None)
    num_stack_levels = kwargs.get('num_stack_levels', 1)
    num_bag_folds = kwargs.get('num_bag_folds', 5)
    num_trails = kwargs.get('num_trails', 5)
    # num_cpus = kwargs.get('num_cpus', 8)

    hyperparameter_tune_kwargs = {
        'num_trials': num_trails,
        'searcher': 'auto',
        'scheduler': 'local',
    }

    cv_seed = kwargs.get('cv_seed', 42)
    cv_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=cv_seed)
    train_index, test_index = next(cv_split.split(data_df_RPs, data_df_RPs['cancer_subtype']))
    train_df = data_df_RPs.iloc[train_index]
    test_df = data_df_RPs.iloc[test_index]

    predictor = TabularPredictor(
            label='cancer_subtype', problem_type=problem_type, eval_metric='accuracy',
            path=f"{model_tmp_directory}/rs{kwargs.get('cv_seed', 42)}"
        ).fit(
            train_data=train_df,
            time_limit=None,
            presets='good_quality',
            num_stack_levels=num_stack_levels,
            num_bag_folds=num_bag_folds,
            auto_stack=False,
            # hyperparameters=hyperparameter_tune_kwargs
        )
    
    # Evaluate the model
    y_true = test_df['cancer_subtype']
    y_probs = predictor.predict_proba(test_df)
    print("y_probs:", y_probs)
    metrics = evaluate_models_predict(y_true, y_probs[:, 1])
    return metrics


def RPSLearner(X, y, **kwargs):
    if kwargs.get('log_transform', True):
        print("Applying log-transform...")
        X = np.log1p(X)

    if kwargs.get('normalize', False):
        print("Normalizing data...")
        X = data_normalization(X)

    label_mapping = {"LUAD": 0, "LUSC": 1}
    y = y.map(label_mapping)
    y = y.values

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=kwargs.get('cv_seed', 42))
    splits = list(cv_strategy.split(X, y))

    # Process folds in parallel
    print("Starting parallel processing of folds...")
    results = Parallel(n_jobs=kwargs.get('n_jobs', 5))(delayed(process_fold)(
        fold_id, train_index, test_index, X, y, **kwargs
    ) for fold_id, (train_index, test_index) in enumerate(splits))

    # Collect y_true and y_probs
    y_true = np.concatenate([res[0] for res in results])
    y_probs = np.concatenate([res[1] for res in results])

    # Compute metrics
    print("Evaluating model performance...")
    metrics = evaluate_models_predict(y_true, y_probs)

    print("Metrics:", metrics)
    return metrics
