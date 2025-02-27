import streamlit as st
import polars as pl
import numpy as np
from classification import find_neighbors
from visualize_data import fig


embeddings_df = pl.read_csv("data/full_embeddings.csv")


st.title("Análise de Embeddings e Classificação")

st.plotly_chart(fig)

selected_index = st.selectbox("Escolha um índice para buscar vizinhos:", range(len(embeddings_df)))
query_embedding = np.array(embeddings_df["embedding"][selected_index])

neighbors = find_neighbors(query_embedding, k=5)
st.write("Top 5 vizinhos encontrados:", neighbors)
