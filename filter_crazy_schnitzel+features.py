import pandas as pd
import numpy as np

# ============================================================================
# 1. ÎNCĂRCARE ȘI FILTRARE DATE
# ============================================================================

# Citire dataset
df = pd.read_csv('dataset-modified.csv')

# Filtrare bonuri care conțin Crazy Schnitzel
bonuri_cu_crazy_schnitzel = df[df['retail_product_name'] == 'Crazy Schnitzel']['id_bon'].unique()
df_crazy_schnitzel = df[df['id_bon'].isin(bonuri_cu_crazy_schnitzel)].copy()
df_crazy_schnitzel = df_crazy_schnitzel.sort_values(['id_bon', 'data_bon'])

# Creare coloană target: y=1 dacă bonul conține Crazy Sauce, altfel 0
bonuri_cu_crazy_sauce = df_crazy_schnitzel[df_crazy_schnitzel['retail_product_name'] == 'Crazy Sauce']['id_bon'].unique()
df_crazy_schnitzel['y'] = df_crazy_schnitzel['id_bon'].apply(lambda x: 1 if x in bonuri_cu_crazy_sauce else 0)

# Salvare dataset filtrat
df_crazy_schnitzel.to_csv('dataset-crazy-schnitzel.csv', index=False)


# ============================================================================
# 2. CREARE FEATURES LA NIVEL DE BON
# ============================================================================

# Procesare date temporale
df_crazy_schnitzel['data_bon'] = pd.to_datetime(df_crazy_schnitzel['data_bon'])
df_crazy_schnitzel['day_of_week'] = df_crazy_schnitzel['data_bon'].dt.dayofweek + 1
df_crazy_schnitzel['is_weekend'] = (df_crazy_schnitzel['data_bon'].dt.dayofweek >= 4).astype(int)
df_crazy_schnitzel['hour'] = df_crazy_schnitzel['data_bon'].dt.hour

# Agregare la nivel de bon
bon_features_list = []

for id_bon, group in df_crazy_schnitzel.groupby('id_bon'):
    features = {
        'id_bon': id_bon,
        'y': group['y'].iloc[0],
        
       
        'day_of_week': group['day_of_week'].iloc[0],
        'is_weekend': group['is_weekend'].iloc[0],
        'hour': group['hour'].iloc[0],
        'is_morning': int(group['perioada_zilei'].iloc[0] == 'morning'),
        'is_lunch': int(group['perioada_zilei'].iloc[0] == 'lunch'),
        'is_evening': int(group['perioada_zilei'].iloc[0] == 'evening'),
        
        'cart_size': len(group),
        'distinct_products': group['retail_product_name'].nunique(),
        'total_value': group['SalePriceWithVAT'].sum(),
        'avg_price': group['SalePriceWithVAT'].mean()
    }
    

    if 'product_category' in group.columns:
        all_categories = ['packaging', 'schnitzel', 'drink', 'side_dish', 'main_dish', 'extra', 'salad', 'dessert']
        for category in all_categories:
            cat_products = group[group['product_category'] == category]
            features[f'count_{category}'] = len(cat_products)
            features[f'value_{category}'] = cat_products['SalePriceWithVAT'].sum()
    
    
    for product_name in group['retail_product_name'].unique():
        if product_name != 'Crazy Sauce':  
            product_count = len(group[group['retail_product_name'] == product_name])
            safe_name = product_name.replace(' ', '_').replace('-', '_').replace('&', 'and').lower()
            features[f'has_{safe_name}'] = product_count
    
    bon_features_list.append(features)
    
df_bon_features = pd.DataFrame(bon_features_list)

df_bon_features.to_csv('dataset-bon-features.csv', index=False)
