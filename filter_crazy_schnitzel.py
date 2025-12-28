import pandas as pd


df = pd.read_csv('dataset-modified.csv')

bonuri_cu_crazy_schnitzel = df[df['retail_product_name'] == 'Crazy Schnitzel']['id_bon'].unique()

df_crazy_schnitzel = df[df['id_bon'].isin(bonuri_cu_crazy_schnitzel)]
df_crazy_schnitzel = df_crazy_schnitzel.sort_values(['id_bon', 'data_bon'])
bonuri_cu_crazy_sauce = df_crazy_schnitzel[df_crazy_schnitzel['retail_product_name'] == 'Crazy Sauce']['id_bon'].unique()
df_crazy_schnitzel['y'] = df_crazy_schnitzel['id_bon'].apply(lambda x: 1 if x in bonuri_cu_crazy_sauce else 0)

df_crazy_schnitzel.to_csv('dataset-crazy-schnitzel.csv', index=False)

print(f"Total bonuri: {df_crazy_schnitzel['id_bon'].nunique()}")
print(f"Bonuri cu Crazy Sauce (y=1): {df_crazy_schnitzel[df_crazy_schnitzel['y']==1]['id_bon'].nunique()}")
print(f"Bonuri fără Crazy Sauce (y=0): {df_crazy_schnitzel[df_crazy_schnitzel['y']==0]['id_bon'].nunique()}")
