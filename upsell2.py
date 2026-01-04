import pandas as pd
import numpy as np
import math
from collections import defaultdict, Counter
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.model_selection import train_test_split


df = pd.read_csv("dataset-original.csv")
df["data_bon"] = pd.to_datetime(df["data_bon"])

df["tip_zi"] = df["data_bon"].dt.weekday.apply(
    lambda x: "weekend" if x >= 5 else "weekday"
)

def perioada(h):
    if 6 <= h < 11: return "dimineata"
    if 11 <= h < 16: return "pranz"
    if 16 <= h < 22: return "seara"
    return "noapte"

df["perioada"] = df["data_bon"].dt.hour.apply(perioada)


UPS_KEYWORDS = [
    "cola","fanta","sprite","aqua","apa","juice",
    "sos","sauce","ketchup","maioneza",
    "cartofi","fries","side","salata"
]

df = df[
    df["retail_product_name"]
    .str.lower()
    .str.contains("|".join(UPS_KEYWORDS), regex=True)
]


bons = []
for bon_id, g in df.groupby("id_bon"):
    if len(g) < 2:
        continue
    bons.append({
        "id_bon": bon_id,
        "products": list(g["retail_product_name"]),
        "tip_zi": g["tip_zi"].iloc[0],
        "perioada": g["perioada"].iloc[0]
    })

bons = pd.DataFrame(bons)
train_bons, test_bons = train_test_split(bons, test_size=0.2, random_state=42)


def generate_samples(bons_df):
    samples = []
    for _, row in bons_df.iterrows():
        for removed in row["products"]:
            basket = [p for p in row["products"] if p != removed]
            samples.append({
                "basket": basket,
                "tip_zi": row["tip_zi"],
                "perioada": row["perioada"],
                "target": removed
            })
    return pd.DataFrame(samples)

train_samples = generate_samples(train_bons)
test_samples  = generate_samples(test_bons)


mlb = MultiLabelBinarizer()
X_train_basket = mlb.fit_transform(train_samples["basket"])
X_test_basket  = mlb.transform(test_samples["basket"])

enc = OneHotEncoder()
X_train_ctx = enc.fit_transform(train_samples[["tip_zi","perioada"]]).toarray()
X_test_ctx  = enc.transform(test_samples[["tip_zi","perioada"]]).toarray()

X_train_knn = np.hstack([X_train_basket, X_train_ctx])
X_test_knn  = np.hstack([X_test_basket, X_test_ctx])

y_train = train_samples["target"].values
y_test  = test_samples["target"].values


train_df = df[df["id_bon"].isin(train_bons["id_bon"])]

prices = train_df.groupby("retail_product_name")["SalePriceWithVAT"].mean().to_dict()
popularity = train_df["retail_product_name"].value_counts().to_dict()
products = list(popularity.keys())
rev_total = train_df.groupby("retail_product_name")["SalePriceWithVAT"].sum().to_dict()

cooc = defaultdict(Counter)
for b in train_samples["basket"]:
    for p in b:
        for q in b:
            if p != q:
                cooc[p][q] += 1

def generate_candidates(basket, top_n=25):
    cands = Counter()
    for p in basket:
        cands.update(cooc[p])
    return [p for p,_ in cands.most_common(top_n)]

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
            for f in basket:
                self.likelihood[prod][f] += 1
            for f in ctx:
                self.likelihood[prod][f] += 0.3

        for p in self.products:
            denom = sum(self.likelihood[p].values()) + len(self.likelihood[p])
            for f in self.likelihood[p]:
                self.likelihood[p][f] = (self.likelihood[p][f] + 5) / (denom + 5*len(self.likelihood[p]))

    def predict_proba(self, basket, ctx):
        scores = {}
        for p in self.products:
            logp = math.log(self.prior.get(p,1e-6))
            for f in basket:
                logp += math.log(self.likelihood[p].get(f,1e-6))
            for f in ctx:
                logp += 0.3 * math.log(self.likelihood[p].get(f,1e-6))
            scores[p] = math.exp(logp)
        s = sum(scores.values())
        return {k:v/s for k,v in scores.items()} if s>0 else scores

class KNNUpsellVector:
    def __init__(self, k=15):
        self.k = k

    def fit(self, X, y):
        self.X = X.astype(bool)
        self.y = y

    def predict_proba(self, x):
        x = x.astype(bool)
        inter = np.sum(self.X & x, axis=1)
        union = np.sum(self.X | x, axis=1)
        sim = inter / (union + 1e-6)

        idx = np.argsort(sim)[-self.k:]
        scores = defaultdict(float)
        for i in idx:
            scores[self.y[i]] += sim[i]**3

        s = sum(scores.values())
        return {k:v/s for k,v in scores.items()} if s>0 else scores

class ID3Node:
    def __init__(self, depth=0, max_depth=6):
        self.depth = depth
        self.max_depth = max_depth
        self.feature = None
        self.children = {}
        self.probs = {}

    def fit(self, baskets, targets):
        if len(set(targets)) == 1 or self.depth >= self.max_depth:
            c = Counter(targets)
            s = sum(c.values())
            self.probs = {k:v/s for k,v in c.items()}
            return

        base_entropy = self.entropy(targets)
        best_gain, best_feature = -1, None

        features = set(p for b in baskets for p in b)
        for f in features:
            left = [targets[i] for i,b in enumerate(baskets) if f in b]
            right = [targets[i] for i,b in enumerate(baskets) if f not in b]
            if not left or not right:
                continue
            gain = base_entropy - (
                len(left)/len(targets)*self.entropy(left) +
                len(right)/len(targets)*self.entropy(right)
            )
            if gain > best_gain:
                best_gain, best_feature = gain, f

        if best_feature is None:
            c = Counter(targets)
            s = sum(c.values())
            self.probs = {k:v/s for k,v in c.items()}
            return

        self.feature = best_feature
        self.children["yes"] = ID3Node(self.depth+1, self.max_depth)
        self.children["no"]  = ID3Node(self.depth+1, self.max_depth)

        left_idx = [i for i,b in enumerate(baskets) if best_feature in b]
        right_idx = [i for i,b in enumerate(baskets) if best_feature not in b]

        self.children["yes"].fit([baskets[i] for i in left_idx], [targets[i] for i in left_idx])
        self.children["no"].fit([baskets[i] for i in right_idx], [targets[i] for i in right_idx])

    def entropy(self, y):
        c = Counter(y)
        s = sum(c.values())
        return -sum((v/s)*math.log2(v/s) for v in c.values())

    def predict_proba(self, basket):
        if self.feature is None:
            return self.probs
        return self.children["yes" if self.feature in basket else "no"].predict_proba(basket)

class AdaBoostUpsell:
    def __init__(self, n_estimators=100):
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []

    def fit(self, baskets, y):
        n = len(y)
        w = np.ones(n)/n
        K = len(set(y))

        for _ in range(self.n_estimators):
            stump = ID3Node(max_depth=3)
            stump.fit(baskets, y)

            preds = [max(stump.predict_proba(b).items(), key=lambda x:x[1])[0] for b in baskets]
            err = np.sum(w * (np.array(preds) != y))
            if err >= 1 - 1/K:
                continue

            alpha = math.log((1-err)/max(err,1e-6)) + math.log(K-1)
            self.models.append(stump)
            self.alphas.append(alpha)

            w *= np.exp(alpha * (np.array(preds) != y))
            w /= w.sum()

    def predict_proba(self, basket):
        scores = defaultdict(float)
        for stump, alpha in zip(self.models, self.alphas):
            for k,v in stump.predict_proba(basket).items():
                scores[k] += alpha * v
        s = sum(scores.values())
        return {k:v/s for k,v in scores.items()} if s>0 else scores


def rank_from_proba(proba, basket):
    candidates = generate_candidates(basket)
    if not candidates:
        candidates = products
    return sorted(
        candidates,
        key=lambda p: (proba.get(p,0)**0.85) *
                      (prices.get(p,1)**0.1) *
                      (popularity.get(p,1)**0.05),
        reverse=True
    )

def hit_at_k(rank, target, k):
    return int(target in rank[:k])


nb = NaiveBayesUpsell()
nb.fit(train_samples["basket"].tolist(),
       train_samples[["tip_zi","perioada"]].values.tolist(),
       y_train)

knn = KNNUpsellVector()
knn.fit(X_train_knn, y_train)

id3 = ID3Node()
id3.fit(train_samples["basket"].tolist(), y_train)

ada = AdaBoostUpsell()
ada.fit(train_samples["basket"].tolist(), y_train)


Ks = [1,3,5]
results = {m:{k:0 for k in Ks} for m in
           ["Popularity","Revenue","NaiveBayes","KNN","ID3","AdaBoost"]}

for _,row in test_samples.iterrows():
    basket = row["basket"]
    ctx = [row["tip_zi"],row["perioada"]]
    target = row["target"]

    pop_rank = sorted(products, key=lambda p: popularity.get(p,0), reverse=True)
    rev_rank = sorted(products, key=lambda p: rev_total.get(p,0), reverse=True)

    nb_rank  = rank_from_proba(nb.predict_proba(basket,ctx), basket)

    ctx_vec = enc.transform(pd.DataFrame([[ctx[0],ctx[1]]], columns=["tip_zi","perioada"])).toarray()
    basket_vec = mlb.transform([basket])
    x_knn = np.hstack([basket_vec, ctx_vec])
    knn_rank = rank_from_proba(knn.predict_proba(x_knn[0]), basket)

    id3_rank = rank_from_proba(id3.predict_proba(basket), basket)
    ada_rank = rank_from_proba(ada.predict_proba(basket), basket)

    for k in Ks:
        results["Popularity"][k] += hit_at_k(pop_rank,target,k)
        results["Revenue"][k] += hit_at_k(rev_rank,target,k)
        results["NaiveBayes"][k] += hit_at_k(nb_rank,target,k)
        results["KNN"][k] += hit_at_k(knn_rank,target,k)
        results["ID3"][k] += hit_at_k(id3_rank,target,k)
        results["AdaBoost"][k] += hit_at_k(ada_rank,target,k)

N = len(test_samples)
for m in results:
    for k in Ks:
        results[m][k] /= N

print(pd.DataFrame(results).T)
