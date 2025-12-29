import pandas as pd
import numpy as np
import math
from collections import defaultdict

from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset-modified.csv")

UPS_CAT = ["drink", "sauce", "side_dish"]
df = df[df["product_category"].isin(UPS_CAT)]

prices = df.groupby("retail_product_name")["SalePriceWithVAT"].mean().to_dict()
products = df["retail_product_name"].unique().tolist()


bons = []

for bon_id, g in df.groupby("id_bon"):
    bons.append({
        "id_bon": bon_id,
        "products": list(g["retail_product_name"]),
        "tip_zi": g["tip_zi"].iloc[0],
        "perioada": g["perioada_zilei"].iloc[0]
    })

bons = pd.DataFrame(bons)

train_bons, test_bons = train_test_split(
    bons,
    test_size=0.2,
    random_state=42
)

def generate_samples(bons_df):
    samples = []
    for _, row in bons_df.iterrows():
        if len(row["products"]) < 2:
            continue

        for removed in row["products"]:
            partial = [p for p in row["products"] if p != removed]

            samples.append({
                "basket": partial,
                "tip_zi": row["tip_zi"],
                "perioada": row["perioada"],
                "target": removed
            })
    return pd.DataFrame(samples)

train_samples = generate_samples(train_bons)
test_samples = generate_samples(test_bons)

mlb = MultiLabelBinarizer()
X_train_basket = mlb.fit_transform(train_samples["basket"])
X_test_basket = mlb.transform(test_samples["basket"])

enc = OneHotEncoder()
X_train_ctx = enc.fit_transform(train_samples[["tip_zi", "perioada"]]).toarray()
X_test_ctx = enc.transform(test_samples[["tip_zi", "perioada"]]).toarray()


X_train = np.hstack([X_train_basket, X_train_ctx])
X_test = np.hstack([X_test_basket, X_test_ctx])

y_train = train_samples["target"].values
y_test = test_samples["target"].values

popularity = df["retail_product_name"].value_counts().to_dict()
revenue = df.groupby("retail_product_name")["SalePriceWithVAT"].sum().to_dict()

def rank_popularity():
    return sorted(products, key=lambda p: popularity.get(p, 0), reverse=True)

def rank_revenue():
    return sorted(products, key=lambda p: revenue.get(p, 0), reverse=True)

class NaiveBayesUpsell:
    def __init__(self):
        self.prior = {}
        self.likelihood = defaultdict(lambda: defaultdict(float))
        self.products = set()

    def fit(self, baskets, contexts, y):
        self.products = set(y)
        total = len(y)

        for p in self.products:
            self.prior[p] = np.sum(y == p) / total

        for basket, ctx, prod in zip(baskets, contexts, y):
            for f in basket + list(ctx):
                self.likelihood[prod][f] += 1

        for p in self.products:
            denom = sum(self.likelihood[p].values()) + len(self.likelihood[p])
            for f in self.likelihood[p]:
                self.likelihood[p][f] = (self.likelihood[p][f] + 1) / denom

    def predict_proba(self, basket, ctx):
        scores = {}
        for p in self.products:
            logp = math.log(self.prior[p])
            for f in basket + list(ctx):
                logp += math.log(self.likelihood[p].get(f, 1e-6))
            scores[p] = math.exp(logp)
        return scores

knn = KNeighborsClassifier(n_neighbors=5)
id3 = DecisionTreeClassifier(criterion="entropy", max_depth=10)
ada = AdaBoostClassifier(n_estimators=100)

knn.fit(X_train, y_train)
id3.fit(X_train, y_train)
ada.fit(X_train, y_train)

nb = NaiveBayesUpsell()
nb.fit(
    train_samples["basket"].tolist(),
    train_samples[["tip_zi", "perioada"]].values.tolist(),
    y_train
)

def rank_from_proba(proba_dict):
    return sorted(
        proba_dict.keys(),
        key=lambda p: proba_dict[p] * prices.get(p, 1),
        reverse=True
    )

def hit_at_k(ranking, target, k):
    return int(target in ranking[:k])

Ks = [1, 3, 5]

results = {
    "Popularity": {k: 0 for k in Ks},
    "Revenue": {k: 0 for k in Ks},
    "NaiveBayes": {k: 0 for k in Ks},
    "KNN": {k: 0 for k in Ks},
    "ID3": {k: 0 for k in Ks},
    "AdaBoost": {k: 0 for k in Ks},
}

for i, row in test_samples.iterrows():
    basket = row["basket"]
    ctx = [row["tip_zi"], row["perioada"]]
    target = row["target"]

    pop_rank = rank_popularity()
    rev_rank = rank_revenue()
    nb_rank = rank_from_proba(nb.predict_proba(basket, ctx))

    x_vec = np.hstack([
    mlb.transform([basket]),
    enc.transform([[row["tip_zi"], row["perioada"]]]).toarray()])


    knn_rank = rank_from_proba(dict(zip(knn.classes_, knn.predict_proba(x_vec)[0])))
    id3_rank = rank_from_proba(dict(zip(id3.classes_, id3.predict_proba(x_vec)[0])))
    ada_rank = rank_from_proba(dict(zip(ada.classes_, ada.predict_proba(x_vec)[0])))

    for k in Ks:
        results["Popularity"][k] += hit_at_k(pop_rank, target, k)
        results["Revenue"][k] += hit_at_k(rev_rank, target, k)
        results["NaiveBayes"][k] += hit_at_k(nb_rank, target, k)
        results["KNN"][k] += hit_at_k(knn_rank, target, k)
        results["ID3"][k] += hit_at_k(id3_rank, target, k)
        results["AdaBoost"][k] += hit_at_k(ada_rank, target, k)

# Normalizare
N = len(test_samples)
for model in results:
    for k in Ks:
        results[model][k] /= N

print(pd.DataFrame(results).T)
