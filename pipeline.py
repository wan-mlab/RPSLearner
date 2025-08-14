import numpy as np
import pandas as pd
import multiprocessing as mp

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.feature_selection import SelectFromModel, SelectKBest

# from autogluon.tabular import TabularPredictor

from model import RandomProjectionReducer, SimpleNNClassifier, MCFS, MetaStacker
from model import get_lasso_model, get_rf_model, get_lightGBM_model, get_xgb_model, get_svm_model
from model import get_l1_logistic_regression_model, get_knn_model, get_extra_trees_model, get_catboost_model

# Configuration
data_path = '/common/wanlab/xinchaowu/Lung_cancer_subtypes/TCGA_lung_rna_seq/'
output_directory = '/work/wanlab/xinchaowu/Lung_Cancer_subtypes/output'

# Data Loading
def load_data(subtype_file, tpm_file, data_directory=data_path):
    subtype_df = pd.read_csv(f'{data_directory}/{subtype_file}')
    subtype = subtype_df["cancer_subtype"]
    tpm = pd.read_csv(f'{data_directory}/{tpm_file}', index_col=0)
    return subtype, tpm


def data_normalization(X):
    """
    This function normalizes the data using z-score normalization.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


def evaluate_models(pipeline, X, y, cv):
    scorers = {
        'acc': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted'),
        'recall': make_scorer(recall_score, average='weighted'),
        'f1': make_scorer(f1_score, average='weighted'),
        'mcc': make_scorer(matthews_corrcoef),
        'auc': make_scorer(roc_auc_score, multi_class='ovr', average='weighted')
    }
    scores = {metric: cross_val_score(pipeline, X, y, cv=cv, scoring=scorer, error_score='raise', n_jobs=-1)
              for metric, scorer in scorers.items()}
    return {
            metric: {'mean': np.mean(score), 'std': np.std(score), 'score': score} for metric, score in scores.items()
        }


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


def process_fold(fold, train_idx, test_idx, pipeline, X, y):
    print(f"  Processing fold {fold}...")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Clone the pipeline to ensure each parallel process has its own pipeline instance
    from sklearn.base import clone
    pipeline_clone = clone(pipeline)

    pipeline_clone.fit(X_train, y_train)
    y_pred = pipeline_clone.predict(X_test)

    # Attempt to get predicted probabilities for AUC
    if hasattr(pipeline_clone.named_steps['classifier'], 'predict_proba'):
        y_proba = pipeline_clone.predict_proba(X_test)[:, 1]  # Assuming binary classification
    elif hasattr(pipeline_clone.named_steps['classifier'], 'decision_function'):
        y_proba = pipeline_clone.decision_function(X_test)
    else:
        y_proba = None

    return {
        'y_true': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba
    }

def evaluate_models_with_aggregated_predictions(pipeline, X, y, cv, parallel=True):
    """
    Perform cross-validation, collect all predictions and true labels,
    and compute metrics on the aggregated data.

    Parameters:
        pipeline: scikit-learn Pipeline object
        X: Features
        y: Labels
        cv: Cross-validation strategy

    Returns:
        metrics: dict containing aggregated metrics
    """
    y_true_all = []
    y_pred_all = []
    y_proba_all = []
    test_indices = []

    splits = list(cv.split(X, y))
    # set the number of jobs to the number of folds
    if parallel:
        n_jobs = len(splits)
        from joblib import Parallel, delayed
        
        # Define the parallel processing
        results = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(process_fold)(fold_idx + 1, train_idx, test_idx, pipeline, X, y)
            for fold_idx, (train_idx, test_idx) in enumerate(splits)
        )

        for result in results:
            y_true_all.append(result['y_true'])
            y_pred_all.append(result['y_pred'])
            if result['y_proba'] is not None:
                y_proba_all.append(result['y_proba'])
            else:
                y_proba_all = None  # If any fold lacks probabilities, set to None

        test_indices = [test_idx for _, test_idx in splits]
        # all_test_indices = [test_idx for _, test_idx in splits]
        all_test_indices = np.concatenate(test_indices)
        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)
        if y_proba_all is not None:
            y_proba_all = np.concatenate(y_proba_all)
        
        n_samples = X.shape[0]
        y_true_ordered = np.zeros(n_samples)
        y_pred_ordered = np.zeros(n_samples)
        y_true_ordered[all_test_indices] = y_true_all
        y_pred_ordered[all_test_indices] = y_pred_all
        y_true_all = y_true_ordered
        y_pred_all = y_pred_ordered

        # if y_proba_all is not None:
        #     y_proba_ordered = np.zeros((n_samples, y_proba_all.shape[1]))
        #     y_proba_ordered[all_test_indices] = y_proba_all
        #     y_proba_all = y_proba_ordered

        
    else:
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
            print(f"  Processing fold {fold}...")
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Fit the pipeline on the training data
            pipeline.fit(X_train, y_train)

            # Predict on the test data
            y_pred = pipeline.predict(X_test)
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)

            # Attempt to get predicted probabilities for AUC
            if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
                y_proba = pipeline.predict_proba(X_test)[:, 1]  # Assuming binary classification
                y_proba_all.extend(y_proba)
            elif hasattr(pipeline.named_steps['classifier'], 'decision_function'):
                y_proba = pipeline.decision_function(X_test)
                y_proba_all.extend(y_proba)
            else:
                # If no probability estimates are available, skip AUC calculation
                y_proba_all = None

    # Compute aggregated metrics
    metrics = {}
    metrics['acc'] = accuracy_score(y_true_all, y_pred_all)
    metrics['precision'] = precision_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
    metrics['f1'] = f1_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
    metrics['mcc'] = matthews_corrcoef(y_true_all, y_pred_all)
    
    if y_proba_all is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true_all, y_proba_all, multi_class='ovr', average='weighted')
        except ValueError as e:
            print(f"  Warning: AUC could not be computed. {e}")
            metrics['auc'] = np.nan
    else:
        metrics['auc'] = np.nan  # Indicate that AUC was not computed

    return metrics, y_true_all, y_pred_all, y_proba_all

def train_and_evaluate(X, y, **kwargs):
    """
    Train and evaluate models using different feature selection and classification methods.
    Parameters:
        X: Features
        y: Labels
    """
    # 1. data normalization: z-score normalization, gene rank normalization
    print("Normalizing data...")
    if kwargs.get('normalize', False):
        X = data_normalization(X)

    # log-transform the data
    if kwargs.get('log_transform', True):
        X = np.log1p(X)

    # one-hot encoding for categorical variables
    label_mapping = {"LUAD": 0, "LUSC": 1}
    y = y.map(label_mapping)
    y = y.values
    # 2. feature selection: LASSO, Random Forest, Gradient Boosting, XGBoost, Rondom Projection, PCA, ANOVA
    feature_selectors = [
        ("MCFS", SelectFromModel(
                MCFS(
                    n_clusters=kwargs.get('MCFS_n_clusters', 5),
                    k=kwargs.get('MCFS_k', 10),
                    alpha=kwargs.get('MCFS_alpha', 0.5),
                ), 
            ),
        ),
        ('LASSO', SelectFromModel(get_lasso_model())),
        # ('Random Forest', SelectFromModel(get_rf_model())),
        # ('Gradient Boosting', SelectFromModel(get_gb_model())),
        # ('XGBoost', SelectFromModel(get_xgb_model())),
        # ('PCA', PCA(n_components=kwargs.get('PCA_components', 50))),
        ('ANOVA', SelectKBest(k=kwargs.get('ANOVA_k', 20))),
    ]
    # 3. classification: SVM, Random Forest, neural networks
    classifiers = {
        'SVM': get_svm_model(),
        'Random Forest': get_rf_model(),
        'Neural Network': SimpleNNClassifier(
                    output_dim=kwargs.get('output_dim', 1),
                    epochs=kwargs.get('epochs', 100),
                    batch_size=kwargs.get('batch_size', 32), 
                    learning_rate=kwargs.get('learning_rate', 1e-4)
                ),
    }
    print("Loading models...")
    pipelines = []
    for fs_name, fs in feature_selectors:
        for clf_name in classifiers:
            pipelines.append(
                    (fs_name + ' + ' + clf_name, 
                     Pipeline(
                         [('feature_selection', fs), 
                          ('classifier', classifiers[clf_name])]
                        )
                    )
                )

    # Evaluate each pipeline using cross-validation, metrics include accuracy, precision, recall, F1, MCC, AUC
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=kwargs.get('cv_seed', 42))

    print("Evaluating models...")
    metrics = {}
    all_true = {}
    all_pred = {}
    all_proba = {}

    for name, pipeline in pipelines:
        print(f"Evaluating pipeline: {name}")
        pipeline_metrics, y_true_all, y_pred_all, y_proba_all = evaluate_models_with_aggregated_predictions(pipeline, X, y, cv=cv_strategy)
        metrics[name] = pipeline_metrics
        all_true[name] = y_true_all
        all_pred[name] = y_pred_all
        all_proba[name] = y_proba_all
        print(f'  Metrics for {name}: {pipeline_metrics}')

    return metrics, all_true, all_pred, all_proba


def train_and_evaluate_lasso(X, y, **kwargs):
    if kwargs.get('log_transform', False):
        X = np.log1p(X)

    if kwargs.get('normalize', False):
        print("Normalizing data...")
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    label_mapping = {"LUAD": 0, "LUSC": 1}
    y = pd.Series(y).map(label_mapping).values
    
    clf = get_l1_logistic_regression_model(
        C=kwargs.get('C', 1.0),
        solver=kwargs.get('solver', 'saga'),
        max_iter=kwargs.get('max_iter', 1000)
    )

    pipeline = Pipeline([
        ('classifier', clf)
    ])
    
    # 6. Cross-validation setup
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=kwargs.get('cv_seed', 42))
    
    print("Evaluating L1-Regularized Logistic Regression...")
    metrics, y_true_all, y_pred_all, y_proba_all = evaluate_models_with_aggregated_predictions(
        pipeline, X, y, cv=cv_strategy
    )
    
    return metrics, y_true_all, y_pred_all, y_proba_all


def train_and_evaluate_RP(X, y, **kwargs):
    """
    Train and evaluate models using Random Projection for feature selection.
    Parameters:
        X: Features
        y: Labels
    """
    # 1. data normalization: z-score normalization, gene rank normalization
    if kwargs.get('normalize', False):
        X = data_normalization(X)

    # log-transform the data
    if kwargs.get('log_transform', True):
        X = np.log1p(X)

    # one-hot encoding for categorical variables
    label_mapping = {"LUAD": 0, "LUSC": 1}
    y = y.map(label_mapping)
    y = y.values

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    classifiers = {
        'SVM': get_svm_model(),
        'Random Forest': get_rf_model(),
        'Neural Network': SimpleNNClassifier(
                    output_dim=kwargs.get('output_dim', 1),
                    epochs=kwargs.get('epochs', 100),
                    batch_size=kwargs.get('batch_size', 32), 
                    learning_rate=kwargs.get('learning_rate', 1e-4)
                ),
    }

    n_RP = kwargs.get('n_RP', 10)
    final_models = []
    eval_metrics = {}
    for name, clf in classifiers.items():
        models = []
        for i in range(n_RP):
            RP_transformer = RandomProjectionReducer(
                reduced_dim=kwargs.get('reduced_dim', 50),
                seedn=n_RP,
                SSSE=kwargs.get('SSSE', True)
            )
            pipe = Pipeline(
                [('RP', RP_transformer), 
                 ('classifier', clf)]
            )
            models.append((f"RP + {name} {i}", pipe))

        # Evaluate and select the best classifiers
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        print("Evaluating models...", name)
        scores = [evaluate_models(model[1], X_train, y_train, cv=cv_strategy) for model in models]
        # Select classifiers based on their performance, e.g., top 50%
        top_classifiers_indices = np.argsort([-score['acc']['mean'] for score in scores])[:int(0.5 * n_RP) + 1]
        top_classifiers = [(models[i][0], models[i][1]) for i in top_classifiers_indices]
        # print(f"Selected {len(top_classifiers)} top classifiers for {name}")
        # Create an ensemble of the top classifiers
        # print("Top classifiers:", top_classifiers)
        ensemble_classifier = VotingClassifier(estimators=top_classifiers, voting='soft')
        # print(ensemble_classifier)
        ensemble_classifier.fit(X_train, y_train)
        eval_metrics[name] = [scores[i] for i in top_classifiers_indices]
        final_models.append((f"Ensemble RP + {name}", ensemble_classifier))

    metrics = {}
    print("Evaluating final models...")
    for name, ensemble_classifier in final_models:
        print(f"Predicting... {name}", ensemble_classifier)
        y_pred = ensemble_classifier.predict(X_test)
        y_probas = ensemble_classifier.predict_proba(X_test)
        metrics[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1': f1_score(y_test, y_pred, average='weighted'),
            'MCC': matthews_corrcoef(y_test, y_pred),
            'AUC': roc_auc_score(y_test, y_probas.argmax(axis=1), multi_class='ovr', average='weighted')
        }
        print(f'Metrics for {name}: {metrics[name]}')
    return metrics, eval_metrics

def train_and_evaluate_RP_xgboost(X, y, **kwargs):
    # Train and evaluate models using Random Projection for dim reduction 
    # classification through ensemble XGBoost or other classifiers
    if kwargs.get('normalize', False):
        X = data_normalization(X)

    # log-transform the data
    if kwargs.get('log_transform', True):
        X = np.log1p(X)

    # label encoding
    label_mapping = {"LUAD": 0, "LUSC": 1}
    y = y.map(label_mapping)
    y = y.values

    # split the data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=kwargs.get('cv_seed', 42), stratify=y
    # )
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=kwargs.get('cv_seed', 42))

    # Parameters
    n_RP = kwargs.get('n_RP', 10)
    SSSE = kwargs.get('SSSE', True)
    reduced_dim = kwargs.get('reduced_dim', 100)
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_estimators': kwargs.get('n_estimators', 100),
    }

    random_projections = [RandomProjectionReducer(
        reduced_dim=reduced_dim,
        seedn=i + kwargs.get('random_state', 42),
        SSSE=SSSE
    ) for i in range(n_RP)]

    y_probas = np.zeros((len(y), 2))
    for train_index, test_index in kf.split(X, y):
        xgboost_clf = get_xgb_model(**xgb_params)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # generate random projections and evaluate
        X_RPs = [RP.fit_transform(X_train) for RP in random_projections]
        X_RPs_test = [RP.transform(X_test) for RP in random_projections]

        X_RPs = np.concatenate(X_RPs, axis=1)
        X_RPs_test = np.concatenate(X_RPs_test, axis=1)

        # fit the model
        print("Fitting the model...")
        xgboost_clf.fit(X_RPs, y_train)
        y_pred = xgboost_clf.predict(X_RPs_test)
        y_proba = xgboost_clf.predict_proba(X_RPs_test)

        y_probas[test_index] = y_proba

    metrics = evaluate_models_predict(y, y_probas[:, 1])
    return metrics

def train_and_evaluate_RP_base_classifiers_ensemble(X, y, **kwargs):
    # Train and evaluate models using Random Projection for dim reduction 
    # classification through ensemble XGBoost or other classifiers
    if kwargs.get('normalize', False):
        X = data_normalization(X)

    # log-transform the data
    if kwargs.get('log_transform', True):
        X = np.log1p(X)

    # label encoding
    label_mapping = {"LUAD": 0, "LUSC": 1}
    y = y.map(label_mapping)
    y = y.values

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=kwargs.get('cv_seed', 42), stratify=y
    )

    # Parameters
    n_RP = kwargs.get('n_RP', 10)
    SSSE = kwargs.get('SSSE', True)
    reduced_dim = kwargs.get('reduced_dim', 50)
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_estimators': 100,
    }

    # generate random projections and evaluate
    classifiers = {
        'XGBoost': get_xgb_model(**xgb_params),
        'Random Forest': get_rf_model(),
        'SVM': get_svm_model(),
    }

    pipelines = {}
    scores = {}
    for name, clf in classifiers.items():
        pipelines[name] = []
        scores[name] = []
        for i in range(n_RP):
            RP_transformer = RandomProjectionReducer(
                reduced_dim=reduced_dim,
                seedn=i,
                SSSE=SSSE
            )
            pipe = Pipeline(
                [('RP', RP_transformer), 
                 (f"{name}_{i}", clf)]
            )
            pipelines[name].append(pipe)

    # ensemble
    probs = {}
    y_preds = {}
    metrics = {}
    for name, pipe_list in pipelines.items():
        print("Ensemble", name)
        # get the probabilities
        probs[name] = []
        for pipe in pipe_list:
            pipe.fit(X_train, y_train)
            probabilities = pipe.predict_proba(X_test)
            probs[name].append(probabilities)
        
        # aggregate the probabilities
        avg_probs = np.mean(probs[name], axis=0)
        y_pred = np.argmax(avg_probs, axis=1)
        y_preds[name] = y_pred

        metrics[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1': f1_score(y_test, y_pred, average='weighted'),
            'MCC': matthews_corrcoef(y_test, y_pred),
            'AUC': roc_auc_score(y_test, avg_probs[:, 1], multi_class='ovr', average='weighted')
        }
        print(f'Metrics for {name}: {metrics[name]}')

    return metrics


def train_RP_clf_ensemble_stackings(X, y, **kwargs):
    if kwargs.get('normalize', False):
        X = data_normalization(X)

    # log-transform the data
    if kwargs.get('log_transform', True):
        X = np.log1p(X)

    # label encoding
    label_mapping = {"LUAD": 0, "LUSC": 1}
    y = y.map(label_mapping)
    y = y.values

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=kwargs.get('cv_seed', 42), stratify=y
    )

    # Parameters
    n_RP = kwargs.get('n_RP', 10)
    SSSE = kwargs.get('SSSE', True)
    reduced_dim = kwargs.get('reduced_dim', 50)
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_estimators': 100,
    }

    # generate random projections and evaluate
    classifiers = {
        'XGBoost': get_xgb_model(**xgb_params),
        'Random Forest': get_rf_model(),
        'SVM': get_svm_model(),
    }

    pipelines = {}
    for name, clf in classifiers.items():
        pipelines[name] = []
        # scores[name] = []
        for i in range(n_RP):
            RP_transformer = RandomProjectionReducer(
                reduced_dim=reduced_dim,
                seedn=i,
                SSSE=SSSE
            )
            pipe = Pipeline(
                [('RP', RP_transformer), 
                 (f"{name}_{i}", clf)]
            )
            pipelines[name].append(pipe)

    # ensemble
    probs = []
    for name, pipe_list in pipelines.items():
        print("Ensemble", name)
        # get the probabilities
        for pipe in pipe_list:
            pipe.fit(X_train, y_train)
            prob = pipe.predict_proba(X_train)
            probs.append(prob)
    # convert to numpy array, shape (n_samples, n_classifiers)
    probs = np.array(probs)
    # using meta-stacking to combine the probabilities
    meta_stacker = MetaStacker()
    meta_stacker.fit(probs, y_train)
    # get the final predictions
    y_pred = meta_stacker.predict(X_test)
    y_probas = meta_stacker.predict_proba(X_test)

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1': f1_score(y_test, y_pred, average='weighted'),
        'MCC': matthews_corrcoef(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_probas[:, 1], multi_class='ovr', average='weighted')
    }
    return metrics

def xgboost_ensemble_process_fold(fold, train_index, test_index, X, y, **kwargs):
    print(f"  Processing fold {fold}...")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # ensemble of XGBoost models
    n_estimators = kwargs.get('n_estimators', 10)
    classifiers = [get_xgb_model(random_state = rs) for rs in range(n_estimators)]

    # fit to the training data
    # print("Training individual classifiers...")
    for idx, clf in enumerate(classifiers):
        # print(f"Training classifier {idx + 1}/{n_estimators}...")
        clf.fit(X_train, y_train)

    # Aggregate the predictions
    # print("Aggregating predictions...")
    probabilities = []
    for idx, clf in enumerate(classifiers):
        # print(f"Predicting with classifier {idx + 1}/{n_estimators}...")
        probabilities.append(clf.predict_proba(X_test))

    avg_proba = np.mean(probabilities, axis=0)
    return avg_proba, y_test

def train_and_evaluate_xgboost_ensemble(X, y, **kwargs):
    """
    Train and evaluate models using different feature selection and classification methods.
    Parameters:
        X: Features
        y: Labels
    """
    # log-transform the data
    if kwargs.get('log_transform', True):
        X = np.log1p(X)
    
    if kwargs.get('normalize', False):
        print("Normalizing data...")
        X = data_normalization(X)
    
    # one-hot encoding for categorical variables
    label_mapping = {"LUAD": 0, "LUSC": 1}
    y = y.map(label_mapping)
    y = y.values

    # 5-fold cross-validation
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=kwargs.get('cv_seed', 42))

    from joblib import Parallel, delayed
    results = Parallel(n_jobs=5)(
        delayed(xgboost_ensemble_process_fold)(fold, train_index, test_index, X, y, **kwargs)
        for fold, (train_index, test_index) in enumerate(cv_strategy.split(X, y))
    )

    y_proba_all = [proba for proba, _ in results]
    y_true_all = [true for _, true in results]

    y_proba_all = np.vstack(y_proba_all)
    y_true_all = np.concatenate(y_true_all)

    y_pred_all = np.argmax(np.array(y_proba_all), axis=1)
    

    print("Evaluating models...")
    # 7. Evaluate final predictions
    metrics = {
        'Accuracy': accuracy_score(y_true_all, y_pred_all),
        'Precision': precision_score(y_true_all, y_pred_all, average='weighted'),
        'Recall': recall_score(y_true_all, y_pred_all, average='weighted'),
        'F1 Score': f1_score(y_true_all, y_pred_all, average='weighted'),
        'MCC': matthews_corrcoef(y_true_all, y_pred_all),
        'AUC': roc_auc_score(y_true_all, y_proba_all[:, 1])
    }
    return metrics


def train_and_evaluate_xgboost_ensemble_stackings(X, y, **kwargs):
    """
    Train and evaluate models using ensemble stacking of XGBoost.
    """
    if kwargs.get('log_transform', True):
        X = np.log1p(X)
    
    if kwargs.get('normalize', False):
        X = data_normalization(X)

    # one-hot encoding for categorical variables
    label_mapping = {"LUAD": 0, "LUSC": 1}
    y = y.map(label_mapping)
    y = y.values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=kwargs.get('test_size', 0.2),
        random_state=kwargs.get('cv_seed', 42),
        stratify=y
    )

    n_estimators = kwargs.get('n_estimators', 10)
    stack = StackingClassifier(
        estimators=[(f'xgb_{i}', get_xgb_model(random_state=i)) for i in range(n_estimators)],
        final_estimator=LogisticRegression(),
        stack_method='predict',
        n_jobs=1
    )

    # Train the stacking classifier
    print("Training the stacking classifier...")
    stack.fit(X_train, y_train)

    y_pred = stack.predict(X_test)
    y_probas = stack.predict_proba(X_test)

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1 Score': f1_score(y_test, y_pred, average='weighted'),
        'MCC': matthews_corrcoef(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_probas[:, 1])
    }

    return metrics

def train_and_evaluate_ag_stacking(X, y, **kwargs):
    """
    Train and evaluate models using autoGluon for stacking.
    Parameters:
        X: Features
        y: Labels
    """
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
    time_limit = kwargs.get('time_limit', 3*60)
    num_stack_levels = kwargs.get('num_stack_levels', 1)
    num_bag_folds = kwargs.get('num_bag_folds', 5)
    num_trails = kwargs.get('num_trails', 5)
    # num_cpus = kwargs.get('num_cpus', 8)

    hyperparameter_tune_kwargs = {
        'num_trials': num_trails,
        'searcher': 'auto',
        'scheduler': 'local',
    }

    performances = []
    predictions = np.zeros((len(data_df_RPs), 2))
    # Split the data into training and testing sets
    if concat:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=kwargs.get('cv_seed', 42))

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(data_df_RPs, data_df_RPs['cancer_subtype'])):
            print(f"Training fold {fold_idx+1}/{skf.get_n_splits()}")
            train_data = data_df_RPs.iloc[train_idx]
            test_data = data_df_RPs.iloc[test_idx]

            predictor = TabularPredictor(
                label='cancer_subtype', problem_type=problem_type, eval_metric='accuracy',
                path=f"{model_tmp_directory}/fold_{fold_idx}_rs{kwargs.get('cv_seed', 42)}"
            ).fit(
                train_data=train_data,
                time_limit=None,
                presets='best_quality',
                num_stack_levels=num_stack_levels,
                num_bag_folds=num_bag_folds,
                auto_stack=False,
                # hyperparameter_tune_kwargs={
                #     'num_trials': num_trails,
                #     'searcher': 'auto',
                #     'scheduler': 'local',
                # },
                # num_gpus=1,
            )
            performance = predictor.evaluate(test_data)
            performances.append(performance)
            predictions[test_idx] = predictor.predict_proba(test_data)
    else:
        for idx, df in enumerate(data_df_RPs):
            print(f"Training on Random Projection {idx+1}/{len(data_df_RPs)}")
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=kwargs.get('cv_seed', 42))
            for fold_idx, (train_index, test_index) in enumerate(skf.split(df, df['cancer_subtype'])):
                print(f"  Fold {fold_idx+1}/{skf.get_n_splits()}")
                train_data = df.iloc[train_index]
                test_data = df.iloc[test_index]

                predictor = TabularPredictor(
                    label='cancer_subtype', problem_type=problem_type, eval_metric='accuracy',
                    path=f"{model_tmp_directory}/RP_{idx}_fold_{fold_idx}"
                ).fit(
                    train_data=train_data,
                    time_limit=time_limit,
                    presets='best_quality',
                    num_stack_levels=num_stack_levels,
                    num_bag_folds=num_bag_folds,
                    hyperparameter_tune_kwargs=hyperparameter_tune_kwargs
                )
                performance = predictor.evaluate(test_data)
            performances.append(performance)
            predictions[test_index] = predictor.predict_proba(test_data)

    # Using the K-fold cross-validation results to get the final performance
    label_mapping = {"LUAD": 0, "LUSC": 1}
    y = data_df_RPs['cancer_subtype'].map(label_mapping)
    metrics = evaluate_models_predict(y, predictions[:, 1])
    return metrics, performances


def train_ag_stacking(X, y, **kwargs):
    # Log-transform the data if specified
    if kwargs.get('log_transform', True):
        print("Applying log-transform to the data...")
        X = np.log1p(X)

    # Normalize the data using z-score normalization if specified
    if kwargs.get('normalize', False):
        print("Normalizing the data using z-score normalization...")
        X = data_normalization(X)

    # Reset index and ensure y is a DataFrame with the correct column name
    y_df = y.reset_index(drop=True)
    if isinstance(y_df, pd.Series):
        y_df = y_df.to_frame()
    if 'cancer_subtype' not in y_df.columns:
        y_df.columns = ['cancer_subtype']
    
    # Convert X to DataFrame if it's not already
    if not isinstance(X, pd.DataFrame):
        X_df = pd.DataFrame(X)
    else:
        X_df = X.copy()
    
    # Concatenate features and labels into a single DataFrame
    data_df = pd.concat([X_df, y_df], axis=1)
    
    # Define model directory and other hyperparameters
    model_tmp_directory = kwargs.get('model_tmp_directory', '/work/wanlab/xinchaowu/Lung_Cancer_subtypes/tmp/ag_models')
    num_stack_levels = kwargs.get('num_stack_levels', 1)
    num_bag_folds = kwargs.get('num_bag_folds', 5)
    problem_type = kwargs.get('problem_type', 'binary')

    # Initialize lists to store performance metrics and predictions
    performances = []
    predictions = np.zeros((len(data_df), 2))  # Assuming binary classification (probabilities for two classes)

    # Initialize Stratified K-Fold cross-validator
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=kwargs.get('cv_seed', 42))
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(data_df, data_df['cancer_subtype']), 1):
        print(f"Training fold {fold_idx}/{skf.get_n_splits()}")

        # Split the data into training and testing sets for the current fold
        train_data = data_df.iloc[train_idx]
        test_data = data_df.iloc[test_idx]

        # Initialize the AutoGluon TabularPredictor
        predictor = TabularPredictor(
            label='cancer_subtype',
            problem_type=problem_type,
            eval_metric='accuracy',
            path=f"{model_tmp_directory}/fold_{fold_idx}_rs{kwargs.get('cv_seed', 42)}"
        ).fit(
            train_data=train_data,
            time_limit=None,  # Unlimited time for fitting; adjust as needed
            presets='medium_quality',
            num_stack_levels=num_stack_levels,
            num_bag_folds=num_bag_folds,
            num_gpus=kwargs.get('num_gpus', 0),  # Adjust based on available GPUs
        )

        # Evaluate the model on the test set and store the performance
        performance = predictor.evaluate(test_data)
        performances.append(performance)
        predictions[test_idx] = predictor.predict_proba(test_data)

    label_mapping = {"LUAD": 0, "LUSC": 1}
    y_mapped = data_df['cancer_subtype'].map(label_mapping)
    metrics = evaluate_models_predict(y_mapped, predictions[:, 1])  # Assuming class 1 is 'LUSC'

    return metrics, performances
    
def rp_xgboost_ensemble_fold_processing(train_index, test_index, X, y, kwargs):
    # train_index, test_index, X, y, kwargs = args
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Extract parameters
    n_estimators = kwargs.get('n_estimators', 10)
    n_RP = kwargs.get('n_RP', 10)
    SSSE = kwargs.get('SSSE', True)
    reduced_dim = kwargs.get('reduced_dim', 50)

    # Initialize Random Projections
    random_projections = [
        RandomProjectionReducer(
            reduced_dim=reduced_dim,
            seedn=i,
            SSSE=SSSE
        ) for i in range(n_RP)
    ]

    # Apply Random Projections
    X_train_RPs = np.concatenate([RP.fit_transform(X_train) for RP in random_projections], axis=1)
    X_test_RPs = np.concatenate([RP.transform(X_test) for RP in random_projections], axis=1)

    classifiers = [get_xgb_model(random_state=rs) for rs in range(n_estimators)]

    for clf in classifiers:
        clf.fit(X_train_RPs, y_train)

    # Aggregate predictions
    probabilities = np.array([clf.predict_proba(X_test_RPs) for clf in classifiers])
    avg_proba = np.mean(probabilities, axis=0)
    return avg_proba, y_test


def train_and_evaluate_rp_xgboost_ensemble(X, y, **kwargs):
    # from multiprocessing import Pool
    from joblib import Parallel, delayed
    # log-transform the data
    X = np.log1p(X)
    if kwargs.get('normalize', False):
        X = data_normalization(X)
    # one-hot encoding for categorical variables
    label_mapping = {"LUAD": 0, "LUSC": 1}
    y = y.map(label_mapping)
    y = y.values

    # Using 5-fold cross-validation
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=kwargs.get('cv_seed', 42))

    # args = [
    #     (train_idx, test_idx, X, y, kwargs)
    #     for train_idx, test_idx in cv_strategy.split(X, y)
    # ]

    # with Pool(processes=5) as pool:
    #     results = pool.map(rp_xgboost_ensemble_fold_processing, args)
    splits = list(cv_strategy.split(X, y))

    # Define the delayed function with all required arguments
    delayed_func = delayed(rp_xgboost_ensemble_fold_processing)

    # Execute in parallel
    results = Parallel(n_jobs=5)(
        delayed_func(train_idx, test_idx, X, y, kwargs) for train_idx, test_idx in splits
    )

    y_proba_all = [proba for proba, _ in results]
    y_true_all = [true for _, true in results]

    y_proba_all = np.vstack(y_proba_all)
    y_true_all = np.concatenate(y_true_all)

    y_pred_all = np.argmax(np.array(y_proba_all), axis=1)

    print("Evaluating models...")
    # 7. Evaluate final predictions
    metrics = {
        'Accuracy': accuracy_score(y_true_all, y_pred_all),
        'Precision': precision_score(y_true_all, y_pred_all, average='weighted'),
        'Recall': recall_score(y_true_all, y_pred_all, average='weighted'),
        'F1 Score': f1_score(y_true_all, y_pred_all, average='weighted'),
        'MCC': matthews_corrcoef(y_true_all, y_pred_all),
        'AUC': roc_auc_score(y_true_all, y_proba_all[:, 1])
    }
    return metrics

  
def train_and_evaluate_xgboost(X, y, **kwargs):
    """
    Train and evaluate models using xgboost.
    """
    X = np.log1p(X)
    if kwargs.get('normalize', False):
        print("Normalizing data...")
        X = data_normalization(X)
    # one-hot encoding for categorical variables
    label_mapping = {"LUAD": 0, "LUSC": 1}
    y = y.map(label_mapping)
    y = y.values

    # Using 5-fold cross-validation
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=kwargs.get('cv_seed', 42))

    # xgboost model
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_estimators': kwargs.get('n_estimators', 100),
    }
    xgb_model = get_xgb_model(**xgb_params)

    # fit to the training data
    print("Training xgboost model...")
    y_preds = np.zeros(len(y))
    for train_index, test_index in cv_strategy.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)
        y_preds[test_index] = y_pred

    # evaluate
    print("Evaluating models...")
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1 Score': f1_score(y_test, y_pred, average='weighted'),
        'MCC': matthews_corrcoef(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_pred)
    }

    return metrics


def individual_base_model_evaluation_fold_processing(fold_id, train_index, test_index, X, y, **kwargs):
    print(f"Processing fold {fold_id+1}...")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Apply random projection
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

    input_dim = X_train_transformed.shape[1]

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
        ('nn', SimpleNNClassifier(input_dim=input_dim, output_dim=1, epochs=100, batch_size=64, learning_rate=1e-4)),
    ]

    y_proba_full = {}
    y_true_full = {}
    for name, model in base_models:
        print(f"Training {name}...")
        model.fit(X_train_transformed, y_train)
        # y_pred = model.predict(X_test_transformed)
        y_proba = model.predict_proba(X_test_transformed)
        y_proba_full[name] = y_proba
        y_true_full[name] = y_test

    return y_true_full, y_proba_full

def individual_base_model_evaluation(X, y, **kwargs):
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



    