import pandas as pd
import numpy as np
import math
from collections import defaultdict, Counter
from itertools import combinations
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.model_selection import train_test_split

# --- Date ---
df = pd.read_csv("dataset-modified.csv")
UPS_CAT = ["drink", "sauce", "side_dish"]
df = df[df["product_category"].isin(UPS_CAT)]

prices = df.groupby("retail_product_name")["SalePriceWithVAT"].mean().to_dict()
popularity = df["retail_product_name"].value_counts().to_dict()
products = df["retail_product_name"].unique().tolist()

# --- Construim bonurile ---
bons = []
for bon_id, g in df.groupby("id_bon"):
    bons.append({
        "id_bon": bon_id,
        "products": list(g["retail_product_name"]),
        "tip_zi": g["tip_zi"].iloc[0],
        "perioada": g["perioada_zilei"].iloc[0]
    })
bons = pd.DataFrame(bons)
train_bons, test_bons = train_test_split(bons, test_size=0.2, random_state=42)

# --- Generare eșantioane ---
def generate_samples(bons_df, max_remove=2):
    samples = []
    for _, row in bons_df.iterrows():
        n = len(row["products"])
        if n < 2:
            continue
        for r in range(1, min(max_remove, n)+1):
            for removed_set in combinations(row["products"], r):
                partial = [p for p in row["products"] if p not in removed_set]
                for removed in removed_set:
                    samples.append({
                        "basket": partial,
                        "tip_zi": row["tip_zi"],
                        "perioada": row["perioada"],
                        "target": removed
                    })
    return pd.DataFrame(samples)

train_samples = generate_samples(train_bons)
test_samples = generate_samples(test_bons)

# --- Feature encoding pentru K-NN vectorial ---
mlb = MultiLabelBinarizer()
X_train_basket = mlb.fit_transform(train_samples["basket"])
X_test_basket = mlb.transform(test_samples["basket"])

enc = OneHotEncoder()
X_train_ctx = enc.fit_transform(train_samples[["tip_zi","perioada"]]).toarray()
X_test_ctx = enc.transform(test_samples[["tip_zi","perioada"]]).toarray()

X_train_knn = np.hstack([X_train_basket, X_train_ctx])
X_test_knn = np.hstack([X_test_basket, X_test_ctx])

y_train_knn = train_samples["target"].values
y_test_knn = test_samples["target"].values

# --- Naive Bayes manual ---
class NaiveBayesUpsell:
    def __init__(self):
        self.prior = {}
        self.likelihood = defaultdict(lambda: defaultdict(float))
        self.products = set()
    def fit(self, baskets, contexts, y):
        self.products = set(y)
        total = len(y)
        for p in self.products:
            self.prior[p] = np.sum(y==p)/total
        for basket, ctx, prod in zip(baskets, contexts, y):
            for f in basket + list(ctx):
                self.likelihood[prod][f] += 1
        # Laplace smoothing mai robust
        for p in self.products:
            denom = sum(self.likelihood[p].values()) + len(self.likelihood[p])
            for f in self.likelihood[p]:
                self.likelihood[p][f] = (self.likelihood[p][f]+1)/(denom + 1e-6)
    def predict_proba(self, basket, ctx):
        scores = {}
        for p in self.products:
            logp = math.log(max(self.prior[p],1e-6))
            for f in basket + list(ctx):
                logp += math.log(self.likelihood[p].get(f, 1e-6))
            scores[p] = math.exp(logp)
        # normalizare
        s = sum(scores.values())
        if s>0:
            for k in scores: scores[k]/=s
        return scores

# --- K-NN manual vectorial ---
class KNNUpsellVector:
    def __init__(self, k=10):
        self.k = k
    def fit(self, X, y):
        self.X = X
        self.y = y
    def predict_proba(self, x_vec):
        X_bin = self.X.astype(bool)
        x_bin = x_vec.astype(bool).reshape(1,-1)
        intersection = np.sum(X_bin & x_bin, axis=1)
        union = np.sum(X_bin | x_bin, axis=1)
        sim = intersection / (union + 1e-6)
        top_idx = np.argsort(sim)[-self.k:]
        scores = defaultdict(float)
        for idx in top_idx:
            scores[self.y[idx]] += sim[idx]**2  # accent pe cei mai similari
        total = sum(scores.values())
        if total>0:
            for k in scores: scores[k]/=total
        return scores

# --- ID3 simplificat ---
class ID3Node:
    def __init__(self, depth=0, max_depth=6):
        self.depth = depth
        self.max_depth = max_depth
        self.feature = None
        self.children = {}
        self.probs = {}
    def fit(self, baskets, contexts, targets):
        if len(set(targets))==1 or self.depth>=self.max_depth:
            c = Counter(targets)
            total=sum(c.values())
            self.probs = {k:v/total for k,v in c.items()}
            return
        best_feature, best_gain = None, -1
        all_products=set(p for b in baskets for p in b)
        total_entropy = self.entropy(targets)
        for prod in all_products:
            left=[targets[i] for i,b in enumerate(baskets) if prod in b]
            right=[targets[i] for i,b in enumerate(baskets) if prod not in b]
            if not left or not right: continue
            gain = total_entropy-(len(left)/len(targets)*self.entropy(left)+len(right)/len(targets)*self.entropy(right))
            if gain>best_gain: best_gain,best_feature=gain,prod
        if best_feature is None:
            c=Counter(targets);total=sum(c.values())
            self.probs={k:v/total for k,v in c.items()}; return
        self.feature=best_feature
        left_idx=[i for i,b in enumerate(baskets) if best_feature in b]
        right_idx=[i for i,b in enumerate(baskets) if best_feature not in b]
        self.children['yes']=ID3Node(depth=self.depth+1,max_depth=self.max_depth)
        self.children['no']=ID3Node(depth=self.depth+1,max_depth=self.max_depth)
        self.children['yes'].fit([baskets[i] for i in left_idx],[contexts[i] for i in left_idx],[targets[i] for i in left_idx])
        self.children['no'].fit([baskets[i] for i in right_idx],[contexts[i] for i in right_idx],[targets[i] for i in right_idx])
    def entropy(self, y):
        c=Counter(y);total=sum(c.values())
        return -sum((v/total)*np.log2(v/total) for v in c.values() if v>0)
    def predict_proba(self,basket,ctx):
        if self.feature is None: return self.probs
        branch='yes' if self.feature in basket else 'no'
        return self.children[branch].predict_proba(basket,ctx)

# --- AdaBoost simplificat ---
class AdaBoostUpsell:
    def __init__(self,n_estimators=100):
        self.n_estimators=n_estimators
        self.models=[]
        self.alphas=[]
    def fit(self, baskets, contexts, y):
        n = len(y)
        w = np.ones(n)/n
        classes = list(set(y))
        n_classes = len(classes)
    
        for _ in range(self.n_estimators):
            stump = ID3Node(max_depth=2)
            stump.fit(baskets, contexts, y)
        
            pred = [max(stump.predict_proba(b,c).items(), key=lambda x:x[1])[0] for b,c in zip(baskets, contexts)]
            err = np.sum(w * (np.array(pred)!=np.array(y))) / np.sum(w)
            if err >= 1-1/n_classes:
                continue
            alpha = np.log((1-err)/max(err,1e-10)) + np.log(n_classes-1)
        
            self.models.append(stump)
            self.alphas.append(alpha)
        
            for i in range(n):
                w[i] *= np.exp(alpha * int(pred[i]!=y[i]))
            w /= w.sum()

    def predict_proba(self,basket,ctx):
        scores=defaultdict(float)
        for stump,alpha in zip(self.models,self.alphas):
            prob=stump.predict_proba(basket,ctx)
            for k,v in prob.items(): scores[k]+=alpha*v
        total=sum(scores.values())
        if total>0: 
            for k in scores: scores[k]/=total
        return scores

# --- Helper Ranking cu ponderi ajustabile ---
def rank_from_proba_manual(proba_dict, alpha=0.7, beta=0.2, gamma=0.1):
    return sorted(
        proba_dict.keys(), 
        key=lambda p: (proba_dict[p]**alpha)*(prices.get(p,1)**beta)*(popularity.get(p,1)**gamma),
        reverse=True
    )

def hit_at_k(ranking,target,k): return int(target in ranking[:k])

# --- Antrenare ---
nb=NaiveBayesUpsell()
nb.fit(train_samples["basket"].tolist(), train_samples[["tip_zi","perioada"]].values.tolist(), train_samples["target"].values)

knn=KNNUpsellVector(k=10)
knn.fit(X_train_knn, y_train_knn)

id3=ID3Node(max_depth=6)
id3.fit(train_samples["basket"].tolist(), train_samples[["tip_zi","perioada"]].values.tolist(), train_samples["target"].values)

ada=AdaBoostUpsell(n_estimators=100)
ada.fit(train_samples["basket"].tolist(), train_samples[["tip_zi","perioada"]].values.tolist(), train_samples["target"].values)

# --- Evaluare Hit@K ---
Ks=[1,3,5]
results={m:{k:0 for k in Ks} for m in ["Popularity","Revenue","NaiveBayes","KNN","ID3","AdaBoost"]}

def rank_popularity(): return sorted(products,key=lambda p:popularity.get(p,0),reverse=True)
def rank_revenue(): return sorted(products,key=lambda p:df.groupby("retail_product_name")["SalePriceWithVAT"].sum().to_dict().get(p,0),reverse=True)

for i,row in test_samples.iterrows():
    basket=row["basket"]; ctx=[row["tip_zi"],row["perioada"]]; target=row["target"]
    pop_rank=rank_popularity()
    rev_rank=rank_revenue()
    nb_rank=rank_from_proba_manual(nb.predict_proba(basket,ctx))
    knn_rank=rank_from_proba_manual(knn.predict_proba(np.hstack([mlb.transform([basket]), enc.transform([[ctx[0],ctx[1]]]).toarray()])))
    id3_rank=rank_from_proba_manual(id3.predict_proba(basket,ctx))
    ada_rank=rank_from_proba_manual(ada.predict_proba(basket,ctx))
    for k in Ks:
        results["Popularity"][k]+=hit_at_k(pop_rank,target,k)
        results["Revenue"][k]+=hit_at_k(rev_rank,target,k)
        results["NaiveBayes"][k]+=hit_at_k(nb_rank,target,k)
        results["KNN"][k]+=hit_at_k(knn_rank,target,k)
        results["ID3"][k]+=hit_at_k(id3_rank,target,k)
        results["AdaBoost"][k]+=hit_at_k(ada_rank,target,k)

# Normalizare
N=len(test_samples)
for model in results:
    for k in Ks:
        results[model][k]/=N

print(pd.DataFrame(results).T)
