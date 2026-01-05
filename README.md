# Sauce Recommender - Machine Learning Project

## Instalare

```bash
# Activare environment
venv\Scripts\activate

# Instalare dependințe
pip install -r requirements.txt
```

## Fișiere Script-uri

- **sauce_recommender.py** - 2.2
- **logistic_regression.py** - 2.1
- **evaluation.py** - Funcții pentru evaluare model (accuracy, precision, recall, F1, ROC-AUC) 2.1

## Folosite intermediar pentru procesare csv
- **create_modified_dataset.py** - Procesează dataset-ul original și adaugă coloane
- **add_category_column.py** - Categorizează produsele în categorii (drink, sauce, schnitzel, etc)
- **filter_crazy_schnitzel+features.py** - Filtrează date și calculează caracteristici din bon

## Fișiere Date (CSV)

- **dataset-original.csv** - Dataset brut inițial cu tranzacții
- **dataset-modified.csv** - Dataset după procesare cu coloane:
  - `product_category` - Categoria produsului (drink, sauce, schnitzel, side_dish, main_dish, extra, salad, dessert, packaging)
- **dataset-bon-features.csv** - Dataset final cu caracteristici agregate pe bon (id_bon ca linie = 1 bon)
  - `count_category_*` - Numărul de produse din fiecare categorie pe bon
  - `count_product_*` - Numărul de orice produs apare pe bon
- **coefficients.csv** - Coeficienți modele antrenate (2.1)

## Fișiere Configurare

- **requirements.txt** - Lista dependințe Python
- **README.md** - Acest fișier cu instrucțiuni
