import polars as pl
import pickle


def load_pickle_data(pickle_path):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    return data


pickle_data = load_pickle_data("data/mini_gm_public_v0.1.p")

df = pl.DataFrame({
    "syndrome_id": [item["syndrome_id"] for item in pickle_data],
    "embedding": [item["embedding"] for item in pickle_data]
})

# Salvar em CSV 
df.write_csv("data/full_embeddings.csv")
print("Dados salvos em 'data/full_embeddings.csv'")
