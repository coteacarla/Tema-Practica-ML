import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

def train_logistic_regression(X, y, learning_rate=0.01, n_iterations=15000, regularization=0.1, 
                              early_stopping_patience=10, tol=1e-6, decay_rate=0.0):
    
    np.random.seed(42)
    m, n_features = X.shape
    limit = np.sqrt(1.0 / n_features)
    weights = np.random.uniform(-limit, limit, n_features)
    bias = 0.0
    
    best_loss = float('inf')
    patience_counter = 0
    loss_history = []
    
    beta1 = 0.9 
    beta2 = 0.999 
    epsilon = 1e-8
    
    m_w, v_w = np.zeros_like(weights), np.zeros_like(weights)
    m_b, v_b = 0.0, 0.0
    t = 0 
    
    current_lr = learning_rate

    for iteration in range(n_iterations):
        if decay_rate > 0:
            current_lr = learning_rate / (1 + decay_rate * iteration)

        predictions = forward_pass(X, weights, bias)
        
        predictions_clipped = np.clip(predictions, 1e-15, 1 - 1e-15)
        bce = -np.mean(y * np.log(predictions_clipped) + (1 - y) * np.log(1 - predictions_clipped))
        l2 = (regularization / (2 * m)) * np.dot(weights, weights)
        loss = bce + l2
        loss_history.append(loss)
        
        dw, db = backward_pass(X, y, predictions, weights, regularization)
        
        t += 1
        
        m_w = beta1 * m_w + (1 - beta1) * dw
        v_w = beta2 * v_w + (1 - beta2) * (dw ** 2)
        
        m_b = beta1 * m_b + (1 - beta1) * db
        v_b = beta2 * v_b + (1 - beta2) * (db ** 2)
        
        m_w_hat = m_w / (1 - beta1 ** t)
        v_w_hat = v_w / (1 - beta2 ** t)
        
        m_b_hat = m_b / (1 - beta1 ** t)
        v_b_hat = v_b / (1 - beta2 ** t)
        
        weights -= current_lr * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
        bias -= current_lr * m_b_hat / (np.sqrt(v_b_hat) + epsilon)
        
        if best_loss - loss > tol:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping_patience:
            print(f"Converged early at iteration {iteration}")
            break
            
        if iteration % 1000 == 0:
            print(f"Iter {iteration}: Loss {loss:.5f}")

    print(f"Final loss: {loss:.5f}, Iterations: {iteration + 1}")
    return weights, bias, loss_history

df = pd.read_csv('dataset-bon-features.csv')
X, y = df.drop(['id_bon', 'y'], axis=1).fillna(0), df['y'].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train_scaled = (X_train - mean) / (std + 1e-8)
X_test_scaled = (X_test - mean) / (std + 1e-8)


weights_custom, bias_custom, loss_history = train_logistic_regression(
    X_train_scaled, y_train,
    learning_rate=0.01,
    n_iterations=15000,
    regularization=0.1,
    early_stopping_patience=10
)

plt.figure(figsize=(10, 6))
plt.plot(loss_history, linewidth=2)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss Curve', fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()

y_proba_custom = forward_pass(X_test_scaled, weights_custom, bias_custom)
y_pred_custom = (y_proba_custom >= 0.5).astype(int)

# sklearn baseline
lr_sklearn = SklearnLR(random_state=42, max_iter=15000, C=10.0)
lr_sklearn.fit(X_train_scaled, y_train)
y_pred_sklearn = lr_sklearn.predict(X_test_scaled)
y_proba_sklearn = lr_sklearn.predict_proba(X_test_scaled)[:, 1]


evaluate_and_report(y_train, y_test, y_pred_custom, y_proba_custom, 
                   y_pred_sklearn, y_proba_sklearn, weights_custom, X.columns)

