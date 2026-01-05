import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as SklearnLR
from evaluation import evaluate_and_report


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def forward_pass(X, weights, bias):
    z = np.dot(X, weights) + bias
    return sigmoid(z)


def backward_pass(X, y, predictions, weights, regularization):
    m = X.shape[0]
    error = predictions - y
    dw = np.dot(X.T, error) / m + (regularization / m) * weights
    db = np.mean(error)
    return dw, db


def train_logistic_regression(X, y, learning_rate, n_iterations, regularization, early_stopping_patience):
    #Xavier
    np.random.seed(42)
    n_features = X.shape[1]
    limit = np.sqrt(1.0 / n_features)
    weights = np.random.uniform(-limit, limit, n_features)
    bias = 0.0
    
    best_loss = float('inf')
    patience = 0
    
    for iteration in range(n_iterations):
        predictions = forward_pass(X, weights, bias)
        predictions_clipped = np.clip(predictions, 1e-15, 1 - 1e-15)
        bce = -np.mean(y * np.log(predictions_clipped) + (1 - y) * np.log(1 - predictions_clipped))
        l2 = (regularization / (2 * X.shape[0])) * np.dot(weights, weights)
        loss = bce + l2
        
      
        dw, db = backward_pass(X, y, predictions, weights, regularization)
        
        weights -= learning_rate * dw
        bias -= learning_rate * db
        
        if loss < best_loss:
            best_loss = loss
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping_patience:
                break
    
    print(f"Final loss: {loss:.4f}, Iterations: {iteration + 1}")
    return weights, bias


df = pd.read_csv('dataset-bon-features.csv')
X, y = df.drop(['id_bon', 'y'], axis=1).fillna(0), df['y'].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std


weights_custom, bias_custom = train_logistic_regression(
    X_train_scaled, y_train,
    learning_rate=0.5,
    n_iterations=15000,
    regularization=0.1,
    early_stopping_patience=5
)

y_proba_custom = forward_pass(X_test_scaled, weights_custom, bias_custom)
y_pred_custom = (y_proba_custom >= 0.5).astype(int)

# sklearn baseline
lr_sklearn = SklearnLR(random_state=42, max_iter=15000, C=10.0)
lr_sklearn.fit(X_train_scaled, y_train)
y_pred_sklearn = lr_sklearn.predict(X_test_scaled)
y_proba_sklearn = lr_sklearn.predict_proba(X_test_scaled)[:, 1]


evaluate_and_report(y_train, y_test, y_pred_custom, y_proba_custom, 
                   y_pred_sklearn, y_proba_sklearn, weights_custom, X.columns)

