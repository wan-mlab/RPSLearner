import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from xgboost import XGBClassifier
from numpy.random import default_rng
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from torch.utils.data import DataLoader, TensorDataset

from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
# from sklearn.feature_selection import SelectorMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import kneighbors_graph
from sklearn.linear_model import Lasso, Lars, LogisticRegression


class SimpleNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim=None, output_dim=1, epochs=100, batch_size=32, learning_rate=1e-4):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.criterion = nn.BCELoss()
        self.optimizer = None
        
    def initialize_model(self):
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.output_dim),
            nn.Sigmoid()
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def fit(self, X, y):
        if self.model is None or self.input_dim != X.shape[1]:
            self.input_dim = X.shape[1]  # Update input_dim based on current data
            self.initialize_model()

        self.classes_ = np.unique(y)

        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.device)

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for data, targets in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
        return self

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
        return torch.round(outputs).cpu().numpy().flatten()
    
    def predict_proba(self, X):
        # Return the probability of the positive class
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
        outputs = outputs.cpu().numpy().flatten()
        return np.vstack([1 - outputs, outputs]).T


class RandomProjectionReducer(BaseEstimator, TransformerMixin):
    def __init__(self, reduced_dim=30, seedn=42, SSSE=True):
        self.reduced_dim = reduced_dim
        self.seedn = seedn
        self.SSSE = SSSE

    def fit(self, X, y=None):
        self.random_projections = self._ranM_SSSE(
                X.shape[1], self.reduced_dim, seedn=self.seedn, SSSE=self.SSSE
            )
        return self
    
    def transform(self, X):
        result = X.dot(self.random_projections.T.toarray())
        return result
    
    def _ranM_SSSE(self, gene_dim, projection_dim, seedn, SSSE):
        # Your existing code here
        if SSSE:
            x0 = np.tile(np.arange(1, projection_dim + 1), int(np.ceil(gene_dim / projection_dim)))
            if seedn % 1 == 0:
                np.random.seed(int(seedn))
                S = np.random.choice(x0, size=gene_dim, replace=False)
                x1 = np.random.choice([-1, 1], size=gene_dim, replace=True, p=[0.5, 0.5])
            else:
                S = np.random.choice(x0, size=gene_dim, replace=False)
                x1 = np.random.choice([-1, 1], size=gene_dim, replace=True, p=[0.5, 0.5])
            x = csr_matrix((x1, (S - 1, np.arange(gene_dim))), shape=(projection_dim, gene_dim))
        else:
            s = np.sqrt(gene_dim)
            rng = default_rng(seedn)
            values = np.array([np.sqrt(s), 0, -np.sqrt(s)], dtype=np.float64)
            probs = np.array([1 / (2*s), 1 - 1/s, 1 / (2*s)])
            x0 = rng.choice(values, size=gene_dim * projection_dim, p=probs)
            x = csr_matrix(
                (x0, (np.repeat(np.arange(projection_dim), gene_dim), 
                      np.tile(np.arange(gene_dim), projection_dim))
                ), shape=(projection_dim, gene_dim)
            )
        return x


class MetaStacker(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim=None, output_dim=1, epochs=100, batch_size=32, learning_rate=1e-4, hidden_layers=4):
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.epochs = epochs
            self.batch_size = batch_size
            self.learning_rate = learning_rate
            self.hidden_layers = hidden_layers
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = None
            self.criterion = nn.BCELoss()
            self.optimizer = None

    def initialize_model(self):
        layers = []
        input_size = self.input_dim
        hidden_size = 128
        for _ in range(self.hidden_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
            hidden_size //= 2
            if hidden_size < 16:
                hidden_size = 16

        layers.append(nn.Linear(input_size, self.output_dim))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def fit(self, X, y):
        # check the dim of X
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.model is None or self.input_dim != X.shape[1]:
            self.input_dim = X.shape[1]
            self.initialize_model()

        self.classes_ = np.unique(y)

        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.device)

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for data, targets in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
        return self

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
        return torch.round(outputs).cpu().numpy().flatten()

    def predict_proba(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
        if self.output_dim == 1:
            outputs = outputs.squeeze().cpu().numpy()
            proba = np.vstack([1 - outputs, outputs]).T
        else:
            proba = outputs.cpu().numpy()
        return proba


class MCFS(BaseEstimator):
    """
    Monte Carlo Feature Selection (MCFS) estimator compatible with SelectFromModel.
    """
    def __init__(self, n_clusters=5, k=5, alpha=0.01):
        """
        Initialize the MCFS feature selector.

        Parameters
        ----------
        n_clusters : int, default=5
            Number of clusters to consider in the spectral embedding.
        k : int, default=5
            Number of nearest neighbors for constructing the affinity matrix.
        alpha : float, default=0.01
            Regularization strength for Lasso regression.
        """
        self.n_clusters = n_clusters
        self.k = k
        self.alpha = alpha

    def fit(self, X, y=None):
        """
        Compute the MCFS feature importances.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored, present here for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        n_samples, n_features = X.shape

        # Step 1: Construct the affinity matrix W using k-nearest neighbors
        W = kneighbors_graph(X, n_neighbors=self.k, mode='connectivity', include_self=False)
        W = 0.5 * (W + W.T)  # Make W symmetric

        # Step 2: Compute the normalized graph Laplacian
        D = W.sum(axis=1).A1  # Degree matrix
        D_inv_sqrt = 1 / np.sqrt(D + 1e-10)  # Avoid division by zero
        D_inv_sqrt_mat = diags(D_inv_sqrt)
        W_normalized = D_inv_sqrt_mat @ W @ D_inv_sqrt_mat  # Normalized affinity matrix

        # Step 3: Compute the eigenvalues and eigenvectors
        eigen_values, eigen_vectors = eigsh(W_normalized, k=self.n_clusters + 1, which='LA')
        Y = eigen_vectors[:, 1:self.n_clusters + 1]  # Exclude the first trivial eigenvector

        # Step 4: Solve K L1-regularized regression problems using Lasso
        W_coef = np.zeros((n_features, self.n_clusters))
        for i in range(self.n_clusters):
            clf = Lasso(alpha=self.alpha, max_iter=1000)
            clf.fit(X, Y[:, i])
            W_coef[:, i] = clf.coef_

        # Compute feature importances
        self.feature_importances_ = np.max(np.abs(W_coef), axis=1)

        return self

# Define other models
def get_lasso_model():
    return Lasso(alpha=0.1)

def get_l1_logistic_regression_model(C=1.0, solver='saga', max_iter=10000):
    return LogisticRegression(
        penalty='l1',
        C=C,
        solver=solver,
        max_iter=max_iter,
        random_state=42
    )

def get_rf_model(criterion="gini", n_estimators=100):
    return RandomForestClassifier(
        n_estimators=n_estimators, 
        criterion=criterion,
        random_state=42
    )

def get_gb_model(n_estimators=100):
    return GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)

def get_xgb_model(**kwargs):
    return XGBClassifier(
        n_estimators=kwargs.get('n_estimators', 100),
        max_depth=kwargs.get('max_depth', 3),
        learning_rate=kwargs.get('learning_rate', 0.1),
        objective=kwargs.get('objective', 'binary:logistic'),
        eval_metric=kwargs.get('eval_metric', 'logloss'),
        n_jobs=kwargs.get('n_jobs', -1),
        random_state=kwargs.get('random_state', 42)
    )

def get_svm_model(kernel="rbf"):
    return SVC(kernel=kernel, probability=True)

def get_catboost_model():
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=100,
        verbose=0,
    )

def get_knn_model(weights='uniform', p=2):
    from sklearn.neighbors import KNeighborsClassifier
    return KNeighborsClassifier(
        weights=weights,
        metric='minkowski',
        p=p
    )

def get_lightGBM_model():
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        n_estimators=100,
        random_state=42,
        verbose=-1
    )
    
def get_extra_trees_model(criteria='gini'):
    from sklearn.ensemble import ExtraTreesClassifier
    return ExtraTreesClassifier(
        criterion=criteria,
        n_estimators=100,
        random_state=42
    )