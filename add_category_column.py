import pandas as pd

df = pd.read_csv('dataset-modified.csv')

def categorize_product(product_name):
    drinks = ['Pepsi', 'Aqua', 'Mountain Dew', 'Prigat', '7Up', 'Lipton', 'Mirinda']
    if any(drink in product_name for drink in drinks):
        return 'drink'
    
    main_dishes = ['Chicken Bao Buns', 'Mac & Cheese with Crispy Bacon', 
                   'Mac & Cheese with Jalapeno and Tomato Sauce']
    if product_name in main_dishes:
        return 'main_dish'

    if product_name.startswith('Extra '):
        return 'extra'
    
    standalone_sauces = ['Crazy Sauce', 'Cheddar Sauce', 'Extra Cheddar Sauce', 
                        'Garlic Sauce', 'Tomato Sauce', 'Blueberry Sauce', 
                        'Spicy Sauce', 'Pink Sauce']
    if product_name in standalone_sauces:
        return 'sauce'
    
    side_dishes = ['Mac & cheease', 'Baked potatoes', 'French fries', 
                   'Crazy Fries with Cheddar Sauce', 'Crazy Fries with Parmesan',
                   'Crazy Fries with Cheddar Sauce and bacon',
                   'Potatoes with Cheddar and Bacon Sauce',
                   'Potatoes with Feta', 'Potatoes with Cheddar Sauce',
                   'French Fries with Parmesan']
    if product_name in side_dishes:
        return 'side_dish'
    

    schnitzel_keywords = ['Schnitzel', 'schnitzel']
    if any(keyword in product_name for keyword in schnitzel_keywords):
        return 'schnitzel'

    if 'Salad' in product_name:
        return 'salad'
    
    desserts = ['Crazy peaches', 'Snails with walnuts and meringue']
    if product_name in desserts:
        return 'dessert'

    if product_name == 'Packaging':
        return 'packaging'
    


df['product_category'] = df['retail_product_name'].apply(categorize_product)


df.to_csv('dataset-modified.csv', index=False)


