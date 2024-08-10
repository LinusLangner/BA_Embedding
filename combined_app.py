import streamlit as st
import gensim.downloader as api
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objs as go
from transformers import AutoTokenizer
import random

hf_token = st.secrets["hf_token"]

# Seiteneinstellungen
st.set_page_config(page_title="Wort-Embeddings & Satz-Tokenizer", layout="wide")

# Allgemeine Stilvorgaben für Konsistenz
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

# Abschnitt 1: Wort-Embeddings Visualisierungen
st.title("🔍 Wort-Embeddings Visualisierungen")

# GloVe-Modell nur einmal laden und cachen
@st.cache_resource
def load_model():
    return api.load('glove-wiki-gigaword-50')

# Modell laden
model = load_model()

# Wortgruppen
tier_worte = ["Hund", "Katze", "Löwe", "Elefant", "Vogel", "Fisch", "Pferd", "Tiger", "Wal", "Bär"]
obst_worte = ["Apfel", "Banane", "Kirsche", "Traube", "Orange", "Birne", "Pfirsich", "Pflaume", "Kiwi", "Mango"]
farben_worte = ["Rot", "Blau", "Grün", "Gelb", "Lila", "Rosa", "Orange", "Schwarz", "Weiß", "Braun"]
emotions_worte = ["Glücklich", "Traurig", "Wütend", "Aufgeregt", "Nervös", "Angst", "Freude", "Liebe", "Hass", "Überraschung"]

# Alle Wörter kombinieren
alle_worte = tier_worte + obst_worte + farben_worte + emotions_worte

# Embeddings für alle Wörter abrufen
embeddings = np.array([model[word.lower()] for word in alle_worte])

# 2D PCA
pca_2d = PCA(n_components=2)
reduzierte_embeddings_2d = pca_2d.fit_transform(embeddings)

# 3D PCA
pca_3d = PCA(n_components=3)
reduzierte_embeddings_3d = pca_3d.fit_transform(embeddings)

# 2D Streudiagramm erstellen
fig_2d = go.Figure()

# Punkte für jede Wortgruppe hinzufügen
for i, words in enumerate([tier_worte, obst_worte, farben_worte, emotions_worte]):
    fig_2d.add_trace(go.Scatter(
        x=reduzierte_embeddings_2d[i*10:(i+1)*10, 0],
        y=reduzierte_embeddings_2d[i*10:(i+1)*10, 1],
        mode='markers+text',
        text=words,
        marker=dict(
            size=12,
            color=['green', 'orange', 'blue', 'red'][i]
        ),
        textposition="top center",
        name=f'{["Tier", "Obst", "Farbe", "Emotion"][i]}-bezogene Wörter'
    ))

# 2D Layout aktualisieren
fig_2d.update_layout(
    xaxis_title='PCA 1',
    yaxis_title='PCA 2',
    height=600,
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

# 3D Streudiagramm erstellen
fig_3d = go.Figure()

# Punkte für jede Wortgruppe hinzufügen
for i, words in enumerate([tier_worte, obst_worte, farben_worte, emotions_worte]):
    fig_3d.add_trace(go.Scatter3d(
        x=reduzierte_embeddings_3d[i*10:(i+1)*10, 0],
        y=reduzierte_embeddings_3d[i*10:(i+1)*10, 1],
        z=reduzierte_embeddings_3d[i*10:(i+1)*10, 2],
        mode='markers+text',
        text=words,
        marker=dict(
            size=8,
            color=['green', 'orange', 'blue', 'red'][i]
        ),
        textposition="top center",
        name=f'{["Tier", "Obst", "Farbe", "Emotion"][i]}-bezogene Wörter'
    ))

# 3D Layout aktualisieren
fig_3d.update_layout(
    scene=dict(
        xaxis_title='PCA 1',
        yaxis_title='PCA 2',
        zaxis_title='PCA 3',
    ),
    height=600,
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

# Layout und Anzeige
st.subheader("2D und 3D Wort-Embeddings Visualisierungen")

# Zwei Spalten erstellen
col1, col2 = st.columns(2)

# Diagramme nebeneinander anzeigen
with col1:
    st.write("#### 2D Visualisierung")
    st.plotly_chart(fig_2d, use_container_width=True, config={'displayModeBar': True, 'scrollZoom': True})

with col2:
    st.write("#### 3D Visualisierung")
    st.plotly_chart(fig_3d, use_container_width=True, config={'displayModeBar': True, 'scrollZoom': True})

# Größeren Abstand für klare Trennung hinzufügen
st.markdown("<div style='height: 150px;'></div>", unsafe_allow_html=True)

# Abschnitt 2: Satz-Tokenizer
st.title("📝 Satz-Tokenizer")

# Definieren einer Reihe von subtilen, halbtransparenten Farben
FARBEN = [
    "rgba(255, 99, 71, 0.3)",   # Tomate
    "rgba(255, 165, 0, 0.3)",   # Orange
    "rgba(255, 215, 0, 0.3)",   # Gold
    "rgba(154, 205, 50, 0.3)",  # Gelbgrün
    "rgba(0, 255, 127, 0.3)",   # Frühlinggrün
    "rgba(100, 149, 237, 0.3)", # Kornblumenblau
    "rgba(138, 43, 226, 0.3)",  # Blauviolett
    "rgba(255, 192, 203, 0.3)", # Rosa
]

# Funktion, um eine zufällige Farbe auszuwählen, die keine aufeinanderfolgenden Wiederholungen enthält
def get_random_color(previous_color=None):
    available_colors = [c for c in FARBEN if c != previous_color]
    return random.choice(available_colors)

# Tokenizer initialisieren
@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", use_auth_token=hf_token)

tokenizer = load_tokenizer()

# Drei Spalten erstellen
col1, col2, col3 = st.columns([2, 1, 1])

# Platzhaltertext
platzhalter_text = "Ein verlassener Garten verwilderte. Ein Junge begann, ihn zu pflegen. Blumen wuchsen bald überall."

# Benutzereingabe in der ersten Spalte
with col1:
    user_input = st.text_area("Geben Sie einen Satz oder eine Abfrage ein:", value=platzhalter_text, height=100)

if user_input:
    # Eingabe tokenisieren
    tokens = tokenizer.tokenize(user_input)

    # Tokenisiertes Ergebnis anzeigen
    st.subheader("Tokenisiertes Ergebnis:")

    # Container für Tokens mit Zeilenumbrüchen erstellen
    tokens_html = '<div style="line-height: 1.6; text-align: left; word-break: break-word;">'
    previous_color = None
    new_sentence = True
    for token in tokens:
        color = get_random_color(previous_color)
        cleaned_token = token.replace('▁', ' ').strip()  # '▁' durch Leerzeichen ersetzen und bereinigen
        if cleaned_token:  # Nur nicht-leere Tokens hinzufügen
            if new_sentence:
                tokens_html += '<div style="margin-bottom: 6px;">'
            tokens_html += f'<span style="background-color:{color}; padding:3px 8px; border-radius:4px; margin-right:6px; margin-bottom:6px; display:inline-block; font-size:1.05em;">{cleaned_token}</span>'
            if cleaned_token.endswith('.'):
                tokens_html += '</div><div style="margin-bottom: 18px;"></div>'  # Größerer Abstand nach einem Punkt
                new_sentence = True
            else:
                new_sentence = False
        previous_color = color
    if not new_sentence:
        tokens_html += '</div>'
    tokens_html += '</div>'

    st.markdown(tokens_html, unsafe_allow_html=True)

    # Token-Informationen anzeigen
    st.subheader("Token-Informationen:")
    st.write(f"Anzahl der Tokens: {len(tokens)}")

    # Token-IDs anzeigen
    token_ids = tokenizer.encode(user_input, add_special_tokens=False)
    st.write("Token-IDs:", token_ids)
