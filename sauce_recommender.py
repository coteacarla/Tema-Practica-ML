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


def train_logistic_regression(X, y, learning_rate=0.5, n_iterations=15000, regularization=0.1, early_stopping_patience=5):
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


def train_sauce_models(df_modified, base_feature_cols):
    models = {}
    results = {}
    
    for sauce in SAUCES:
        sauce_col = f'has_{sauce.lower().replace(" ", "_")}'
        sauce_product_col = f'count_product_{sauce.lower().replace(" ", "_")}'
        
        feature_cols = [col for col in base_feature_cols if col != sauce_col and col != sauce_product_col]
        
        X = df_modified[feature_cols].fillna(0)
        y = df_modified[sauce_col].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        X_train_scaled = (X_train - mean) / std
        X_test_scaled = (X_test - mean) / std
        
        weights, bias = train_logistic_regression(
            X_train_scaled.values, y_train,
            learning_rate=0.25, n_iterations=15000,
            regularization=0.1, early_stopping_patience=5
        )
        
        y_proba = forward_pass(X_test_scaled.values, weights, bias)
        y_pred = (y_proba >= 0.5).astype(int)
        
        acc = (y_pred == y_test).mean()
        prec = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_pred == 1) if np.sum(y_pred == 1) > 0 else 0
        rec = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1) if np.sum(y_test == 1) > 0 else 0
        
        models[sauce] = {
            'weights': weights, 'bias': bias,
            'mean': mean.values, 'std': std.values,
            'feature_cols': feature_cols
        }
        
        results[sauce] = {
            'accuracy': acc, 'precision': prec, 'recall': rec,
            'y_test': y_test, 'y_pred': y_pred, 'y_proba': y_proba
        }
    
    print("\nAccuracy Summary:")
    for sauce in SAUCES:
        acc = results[sauce]['accuracy']
        print(f"  {sauce:25s}: {acc:.4f}")
    
    return models, results


def get_model_recommendations(basket_features, models, k=3):
    recommendations = {}
    
    for sauce, model in models.items():
        sauce_col = f'has_{sauce.lower().replace(" ", "_")}'
        
        if basket_features[sauce_col] == 1:
            continue
        
        feature_cols = model['feature_cols']
        X_basket = basket_features[feature_cols].values.reshape(1, -1)
        X_scaled = (X_basket - model['mean']) / model['std']
        
        prob = forward_pass(X_scaled, model['weights'], model['bias'])[0]
        recommendations[sauce] = prob
    
    sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    return sorted_recs[:k]


def get_popularity_recommendations(df_modified, k=3):
    popularity = {}
    for sauce in SAUCES:
        sauce_col = f'has_{sauce.lower().replace(" ", "_")}'
        popularity[sauce] = df_modified[sauce_col].mean()
    
    sorted_pop = sorted(popularity.items(), key=lambda x: x[1], reverse=True)
    return sorted_pop[:k]


def evaluate_models(df_modified, base_feature_cols, models):
    X_dummy = df_modified[base_feature_cols].fillna(0)
    y_dummy = df_modified[f'has_{SAUCES[0].lower().replace(" ", "_")}'].values
    _, _, test_indices, _ = train_test_split(
        range(len(X_dummy)), y_dummy, test_size=0.2, stratify=y_dummy, random_state=42
    )
    
    results_hit_precision = {
        k: {'model': {'hit': [], 'precision': []}, 'popularity': {'hit': [], 'precision': []}}
        for k in K_VALUES
    }
    
    for idx in test_indices:
        receipt = df_modified.iloc[idx].copy()
        
        actual_sauces = set()
        for sauce in SAUCES:
            sauce_col = f'has_{sauce.lower().replace(" ", "_")}'
            if receipt[sauce_col] == 1:
                actual_sauces.add(sauce)
        
       
        if len(actual_sauces) == 0:
            continue
        
        basket_no_sauce = receipt.copy()
        for sauce in SAUCES:
            sauce_col = f'has_{sauce.lower().replace(" ", "_")}'
            basket_no_sauce[sauce_col] = 0
        
       
        model_recs = get_model_recommendations(basket_no_sauce, models, k=5)
        model_rec_sauces = [sauce for sauce, prob in model_recs]
        
        pop_recs = get_popularity_recommendations(df_modified, k=5)
        pop_rec_sauces = [sauce for sauce, pop in pop_recs][:5]
        
      
        for k in K_VALUES:
            hit_model = len(set(model_rec_sauces[:k]) & actual_sauces) > 0
            precision_model = len(set(model_rec_sauces[:k]) & actual_sauces) / k
            results_hit_precision[k]['model']['hit'].append(1 if hit_model else 0)
            results_hit_precision[k]['model']['precision'].append(precision_model)
            
            hit_pop = len(set(pop_rec_sauces[:k]) & actual_sauces) > 0
            precision_pop = len(set(pop_rec_sauces[:k]) & actual_sauces) / k
            results_hit_precision[k]['popularity']['hit'].append(1 if hit_pop else 0)
            results_hit_precision[k]['popularity']['precision'].append(precision_pop)
    
    print("\nModel vs Baseline:")
    for k in K_VALUES:
        hit_model = np.mean(results_hit_precision[k]['model']['hit'])
        hit_pop = np.mean(results_hit_precision[k]['popularity']['hit'])
        prec_model = np.mean(results_hit_precision[k]['model']['precision'])
        prec_pop = np.mean(results_hit_precision[k]['popularity']['precision'])
        
        print(f"Hit@{k}: Model {hit_model:.4f} vs Baseline {hit_pop:.4f} (Δ {(hit_model - hit_pop):+.4f})")
        print(f"Precision@{k}: Model {prec_model:.4f} vs Baseline {prec_pop:.4f} (Δ {(prec_model - prec_pop):+.4f})")


if __name__ == '__main__':
    df_modified, base_feature_cols = load_and_prepare_data()
    models, results = train_sauce_models(df_modified, base_feature_cols)
    evaluate_models(df_modified, base_feature_cols, models)
