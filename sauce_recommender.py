import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

SAUCES = [
    'Crazy Sauce', 'Cheddar Sauce', 'Extra Cheddar Sauce', 'Garlic Sauce',
    'Tomato Sauce', 'Blueberry Sauce', 'Spicy Sauce', 'Pink Sauce'
]
K_VALUES = [1, 3, 5]

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
        
        if iteration % 100 == 0: 
            bce = -np.mean(y * np.log(predictions_clipped) + (1 - y) * np.log(1 - predictions_clipped))
            l2 = (regularization / (2 * m)) * np.dot(weights, weights)
            loss = bce + l2
            
            if best_loss - loss > tol:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    break

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

    return weights, bias

def load_and_prepare_data():
    df_modified = pd.read_csv('dataset-modified-features.csv')
    df_original = pd.read_csv('dataset-original.csv')
    
    for sauce in SAUCES:
        sauce_col = f'has_{sauce.lower().replace(" ", "_")}'
        sauce_dict = (
            df_original.groupby('id_bon')['retail_product_name']
            .apply(lambda x: int(sauce in x.values))
            .to_dict()
        )
        df_modified[sauce_col] = df_modified['id_bon'].map(sauce_dict).fillna(0).astype(int)
    
    base_feature_cols = list(df_modified.drop(['id_bon'], axis=1).columns)
    return df_modified, base_feature_cols

def train_sauce_models(X_train_df, X_test_df, base_feature_cols):
    models = {}
    results = {}
    
    print("Training models...")
    for sauce in SAUCES:
        sauce_col = f'has_{sauce.lower().replace(" ", "_")}'
        sauce_product_col = f'count_product_{sauce.lower().replace(" ", "_")}'
        
        feature_cols = [
            col for col in base_feature_cols 
            if col != sauce_col and col != sauce_product_col
        ]
        
        X_train = X_train_df[feature_cols].fillna(0).values
        y_train = X_train_df[sauce_col].values
        
        X_test = X_test_df[feature_cols].fillna(0).values
        y_test = X_test_df[sauce_col].values
        
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        X_train_scaled = (X_train - mean) / (std + 1e-8)
        X_test_scaled = (X_test - mean) / (std + 1e-8)
        
        weights, bias = train_logistic_regression(
            X_train_scaled, y_train,
            learning_rate=0.01, regularization=0.1
        )
        
        y_proba = forward_pass(X_test_scaled, weights, bias)
        y_pred = (y_proba >= 0.5).astype(int)
        acc = (y_pred == y_test).mean()
        
        models[sauce] = {
            'weights': weights, 'bias': bias,
            'mean': mean, 'std': std,
            'feature_cols': feature_cols
        }
        results[sauce] = acc
        
    print("Individual Model Accuracies:", results)
    return models

def get_model_recommendations(basket_features, models, k=3):
    recommendations = {}
    
    for sauce, model in models.items():
        sauce_col = f'has_{sauce.lower().replace(" ", "_")}'
        if basket_features[sauce_col] == 1: continue
        
        feature_cols = model['feature_cols']
        X_basket = basket_features[feature_cols].values.reshape(1, -1)
        
        X_scaled = (X_basket - model['mean']) / (model['std'] + 1e-8)
        
        prob = forward_pass(X_scaled, model['weights'], model['bias'])[0]
        recommendations[sauce] = prob
    
    return sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:k]

def get_popularity_recommendations(df_train, k=3):
    popularity = {}
    for sauce in SAUCES:
        sauce_col = f'has_{sauce.lower().replace(" ", "_")}'
        popularity[sauce] = df_train[sauce_col].mean()
    return sorted(popularity.items(), key=lambda x: x[1], reverse=True)[:k]

def evaluate_models(df_test, df_train_ref, models):
    print("\nStarting Global Evaluation...")
    
    pop_recs_list = get_popularity_recommendations(df_train_ref, k=5)
    pop_rec_sauces = [sauce for sauce, pop in pop_recs_list]
    
    results_hit_precision = {
        k: {'model': {'hit': [], 'precision': []}, 'popularity': {'hit': [], 'precision': []}}
        for k in K_VALUES
    }
    
    for idx, receipt in df_test.iterrows():
        actual_sauces = {s for s in SAUCES if receipt[f'has_{s.lower().replace(" ", "_")}'] == 1}
        
        if len(actual_sauces) == 0: continue
        
        basket_no_sauce = receipt.copy()
        for sauce in SAUCES:
            basket_no_sauce[f'has_{sauce.lower().replace(" ", "_")}'] = 0
        
        model_recs = get_model_recommendations(basket_no_sauce, models, k=5)
        model_rec_sauces = [sauce for sauce, prob in model_recs]
        
        for k in K_VALUES:
            top_k_model = model_rec_sauces[:k]
            hits = len(set(top_k_model) & actual_sauces)
            results_hit_precision[k]['model']['hit'].append(1 if hits > 0 else 0)
            results_hit_precision[k]['model']['precision'].append(hits / k)
            
            top_k_pop = pop_rec_sauces[:k]
            hits_pop = len(set(top_k_pop) & actual_sauces)
            results_hit_precision[k]['popularity']['hit'].append(1 if hits_pop > 0 else 0)
            results_hit_precision[k]['popularity']['precision'].append(hits_pop / k)

    print("\n--- Final Results (Test Set) ---")
    for k in K_VALUES:
        hit_model = np.mean(results_hit_precision[k]['model']['hit'])
        hit_pop = np.mean(results_hit_precision[k]['popularity']['hit'])
        print(f"Hit@{k}: Model {hit_model:.4f} vs Baseline {hit_pop:.4f} (Δ {hit_model - hit_pop:+.4f})")

if __name__ == '__main__':
    df_modified, base_feature_cols = load_and_prepare_data()
    df_train, df_test = train_test_split(df_modified, test_size=0.2, random_state=42)
    models = train_sauce_models(df_train, df_test, base_feature_cols)
    evaluate_models(df_test, df_train, models)