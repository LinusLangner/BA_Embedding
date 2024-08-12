import streamlit as st
import gensim.downloader as api
import numpy as np
import plotly.graph_objs as go
from transformers import AutoTokenizer
import random

hf_token = st.secrets["hf_token"]

# Page settings
st.set_page_config(page_title="Transformer Visualisierungen", layout="wide")

# General style guidelines for consistency
st.markdown("""
<style>
    .stApp {
        max-width: 100%;
        padding: 0 2rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .st-emotion-cache-16idsys p {
        font-size: 1.2rem;
    }
    .stTextArea textarea {
        font-size: 1rem;
    }
    .stTextArea .st-emotion-cache-16txtl3 {
        font-size: 0.8rem;
        opacity: 0.7;
    }
    h1, h2, h3 {
        color: #333333;
        font-weight: 600;
    }
    h1 {
        font-size: 2rem;
        margin-bottom: 1rem;
    }
    h2 {
        font-size: 1.75rem;
    }
    h3 {
        font-size: 1.5rem;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-size: 1rem;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        margin-top: 1rem;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stMarkdown {
        font-size: 1.1rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# Section 1: Self-Attention Visualizations
st.title("🔍 Self-Attention Visualisierung")

# Tokenizer initialisieren
@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", use_auth_token=hf_token)

tokenizer = load_tokenizer()

# User input
st.subheader("Geben Sie einen Satz ein:")
user_input = st.text_area("Satz eingeben:", value="Der schnelle braune Fuchs springt über den faulen Hund.", height=100)

if user_input:
    # Eingabe tokenisieren
    tokens = tokenizer.tokenize(user_input)
    token_ids = tokenizer.encode(user_input, add_special_tokens=False)
    
    st.subheader("Tokens:")
    st.write(tokens)

    # Self-Attention Simulation
    attention_matrix = np.random.rand(len(tokens), len(tokens))  # Zufällige Matrix zur Veranschaulichung

    fig_attention = go.Figure(data=go.Heatmap(
        z=attention_matrix,
        x=tokens,
        y=tokens,
        colorscale='Viridis'))
    
    fig_attention.update_layout(
        title="Self-Attention Gewichtungen",
        xaxis_title="Tokens",
        yaxis_title="Tokens",
        height=600,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    
    st.plotly_chart(fig_attention, use_container_width=True)

# Section 2: Positionale Kodierung Visualisierung
st.title("📏 Positionale Kodierung Visualisierung")

def positional_encoding(token_index, d_model, max_len=5000):
    pe = np.zeros(d_model)
    for i in range(0, d_model, 2):
        pe[i] = np.sin(token_index / (10000 ** (i / d_model)))
        pe[i + 1] = np.cos(token_index / (10000 ** ((i + 1) / d_model)))
    return pe

# Berechnung der Positionale Kodierungen
positional_encodings = [positional_encoding(i, len(token_ids)) for i in range(len(token_ids))]

fig_pe = go.Figure(data=go.Heatmap(
    z=positional_encodings,
    x=[f"Dim {i+1}" for i in range(len(positional_encodings[0]))],
    y=tokens,
    colorscale='Blues'))

fig_pe.update_layout(
    title="Positionale Kodierungen",
    xaxis_title="Dimensionen",
    yaxis_title="Tokens",
    height=600,
    margin=dict(l=20, r=20, t=30, b=20)
)

st.plotly_chart(fig_pe, use_container_width=True)

# Möglichkeit, die Reihenfolge der Tokens zu ändern
st.subheader("Ändern Sie die Reihenfolge der Wörter:")
tokens_shuffled = st.multiselect('Neue Reihenfolge:', tokens, default=tokens)

if tokens_shuffled:
    # Neuberechnung der Positionale Kodierung für die neue Reihenfolge
    positional_encodings_shuffled = [positional_encoding(i, len(token_ids)) for i in range(len(tokens_shuffled))]

    fig_pe_shuffled = go.Figure(data=go.Heatmap(
        z=positional_encodings_shuffled,
        x=[f"Dim {i+1}" for i in range(len(positional_encodings_shuffled[0]))],
        y=tokens_shuffled,
        colorscale='Blues'))

    fig_pe_shuffled.update_layout(
        title="Positionale Kodierungen (geänderte Reihenfolge)",
        xaxis_title="Dimensionen",
        yaxis_title="Tokens",
        height=600,
        margin=dict(l=20, r=20, t=30, b=20)
    )

    st.plotly_chart(fig_pe_shuffled, use_container_width=True)
