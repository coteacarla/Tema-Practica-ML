import pandas as pd


df = pd.read_csv('dataset-modified.csv')

modified_data = []

for id_bon in df['id_bon'].unique():
    receipt_df = df[df['id_bon'] == id_bon]
    row = {'id_bon': id_bon}
    
    category_counts = receipt_df['product_category'].value_counts().to_dict()
    for category, count in category_counts.items():
        col_name = f'count_category_{category.lower().replace(" ", "_").replace("-", "_")}'
        row[col_name] = count
    
    product_counts = receipt_df['retail_product_name'].value_counts().to_dict()
    for product, count in product_counts.items():
        col_name = f'count_product_{product.lower().replace(" ", "_").replace("-", "_").replace(".", "_")}'
        row[col_name] = count
    
    modified_data.append(row)

df_features = pd.DataFrame(modified_data)
df_features = df_features.fillna(0).astype({col: int for col in df_features.columns if col != 'id_bon'})
df_features.to_csv('dataset-modified-features.csv', index=False)

print("Dataset with counters created: dataset-modified-features.csv")
print(f"Shape: {df_features.shape}")
print(f"Total features: {len(df_features.columns)}")
print(f"\nSample columns: {df_features.columns.tolist()[:20]}")



