import faiss
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score, f1_score


embeddings_df = pl.read_csv("data/full_embeddings.csv")
embeddings = np.stack(embeddings_df["embedding"].to_list())
labels = embeddings_df["syndrome_id"].to_numpy()

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)


def find_neighbors(query_embedding, k=5):
    D, I = index.search(np.array([query_embedding]), k)
    return I[0]


y_true = labels  
y_pred_prob = np.random.rand(len(labels)) 
y_pred = (y_pred_prob > 0.5).astype(int)

auc = roc_auc_score(y_true, y_pred_prob, multi_class="ovr")
f1 = f1_score(y_true, y_pred, average="weighted")

print(f"AUC: {auc:.3f}, F1-Score: {f1:.3f}")
