import polars as pl
import umap
import numpy as np
import plotly.express as px
import streamlit as st


embeddings_df = pl.read_csv("data/full_embeddings.csv")


reducer = umap.UMAP(n_components=2, random_state=42)
embedding_2d = reducer.fit_transform(np.stack(embeddings_df["embedding"].to_list()))

df_plot = pl.DataFrame({
    "x": embedding_2d[:, 0],
    "y": embedding_2d[:, 1],
    "syndrome_id": embeddings_df["syndrome_id"]
})

# Criar gráfico interativo
fig = px.scatter(df_plot.to_pandas(), x="x", y="y", color="syndrome_id", title="Projeção UMAP dos Embeddings")
st.plotly_chart(fig)
