import streamlit as st
import gensim.downloader as api
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objs as go
from transformers import AutoTokenizer
import random
import networkx as nx
import matplotlib.pyplot as plt

hf_token = st.secrets["hf_token"]

# Page settings
st.set_page_config(page_title="Wort-Embeddings & Satz-Tokenizer", layout="wide")

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

# Section 1: Word Embeddings Visualizations
st.title("🔍 Wort-Embeddings Visualisierungen")

# Load GloVe model only once and cache
@st.cache_resource
def load_model():
    return api.load('glove-wiki-gigaword-50')

# Load model
model = load_model()

# Word groups
tier_worte = ["dog", "cat", "lion", "elephant", "bird", "fish", "horse", "tiger", "whale", "bear"]
obst_worte = ["apple", "banana", "cherry", "grape", "orange", "pear", "peach", "plum", "peanut", "mango"]
farben_worte = ["red", "blue", "green", "yellow", "purple", "pink", "orange", "black", "white", "brown"]
emotions_worte = ["happy", "sad", "angry", "excited", "nervous", "fear", "joy", "love", "hate", "surprise"]

# User input for custom words
st.subheader("Fügen Sie Ihre eigenen Wörter hinzu")
user_words = st.text_input("Geben Sie Wörter ein, getrennt durch Kommas:", "")
user_words = [word.strip().lower() for word in user_words.split(',') if word.strip()]

# Combine all words
alle_worte = tier_worte + obst_worte + farben_worte + emotions_worte + user_words

# Get embeddings for all words, only if the word is in the model
embeddings = []
valid_words = []
for word in alle_worte:
    if word.lower() in model:
        embeddings.append(model[word.lower()])
        valid_words.append(word)
    else:
        st.warning(f"Das Wort '{word}' wurde im GloVe-Modell nicht gefunden und wird ignoriert.")

embeddings = np.array(embeddings)

if len(embeddings) == 0:
    st.error("Keines der Wörter wurde im GloVe-Modell gefunden.")
else:
    # 2D and 3D PCA
    pca_2d = PCA(n_components=2)
    reduzierte_embeddings_2d = pca_2d.fit_transform(embeddings)

    pca_3d = PCA(n_components=3)
    reduzierte_embeddings_3d = pca_3d.fit_transform(embeddings)

    # Create 2D scatter plot
    fig_2d = go.Figure()

    # Add points for each word group
    colors = ['green', 'orange', 'blue', 'red', '#00FFFF']  # Cyan for user inputs
    group_names = ["Tier", "Obst", "Farbe", "Emotion", "Benutzer Eingaben"]

    for i, words in enumerate([tier_worte, obst_worte, farben_worte, emotions_worte, user_words]):
        valid_indices = [j for j, word in enumerate(valid_words) if word in words]
        if valid_indices:
            fig_2d.add_trace(go.Scatter(
                x=reduzierte_embeddings_2d[valid_indices, 0],
                y=reduzierte_embeddings_2d[valid_indices, 1],
                mode='markers+text',
                text=[valid_words[j] for j in valid_indices],
                marker=dict(
                    size=12,
                    color=colors[i],
                ),
                textposition="top center",
                name=f'{group_names[i]}-bezogene Wörter' if i < 4 else group_names[i]
            ))

    # Update 2D layout
    fig_2d.update_layout(
        xaxis_title='PCA 1',
        yaxis_title='PCA 2',
        height=600,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    # Create 3D scatter plot
    fig_3d = go.Figure()
    
    # Add points for each word group
    for i, words in enumerate([tier_worte, obst_worte, farben_worte, emotions_worte, user_words]):
        valid_indices = [j for j, word in enumerate(valid_words) if word in words]
        if valid_indices:
            fig_3d.add_trace(go.Scatter3d(
                x=reduzierte_embeddings_3d[valid_indices, 0],
                y=reduzierte_embeddings_3d[valid_indices, 1],
                z=reduzierte_embeddings_3d[valid_indices, 2],
                mode='markers+text',
                text=[valid_words[j] for j in valid_indices],
                marker=dict(
                    size=8,
                    color=colors[i],
                ),
                textposition="top center",
                name=f'{group_names[i]}-bezogene Wörter' if i < 4 else group_names[i]
            ))

    # Update 3D layout
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

    # Layout and display
    st.subheader("2D und 3D Wort-Embeddings Visualisierungen")

    # Create two columns
    col1, col2 = st.columns(2)

    # Display charts side by side
    with col1:
        st.write("#### 2D Visualisierung")
        st.plotly_chart(fig_2d, use_container_width=True, config={'displayModeBar': True, 'scrollZoom': True})

    with col2:
        st.write("#### 3D Visualisierung")
        st.plotly_chart(fig_3d, use_container_width=True, config={'displayModeBar': True, 'scrollZoom': True})

# Add larger space for clear separation
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

    # Self-Attention Visualisierung
    st.title("🔗 Self-Attention Visualisierung")
    
    # Simulation von Self-Attention-Daten
    if user_input:
        tokens = tokenizer.tokenize(user_input)
        attention_matrix = np.random.rand(len(tokens), len(tokens))  # Zufällige Matrix zur Veranschaulichung
    
        # Erstelle ein Netzwerkdiagramm
        G = nx.DiGraph()
        for i, token in enumerate(tokens):
            for j in range(len(tokens)):
                if i != j:
                    G.add_edge(tokens[i], tokens[j], weight=attention_matrix[i][j])
    
        # Position der Knoten bestimmen (hier: zirkulär)
        pos = nx.circular_layout(G)
    
        # Zeichne das Netzwerkdiagramm
        plt.figure(figsize=(10, 10))
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=3000)
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        
        # Kantenzeichnung mit angepasster Transparenz je nach Gewichtung
        edges = nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=20,
                                       edge_color=[G[u][v]['weight'] for u, v in G.edges],
                                       edge_cmap=plt.cm.Blues, width=2, edge_vmin=0, edge_vmax=1, alpha=0.7)
    
        # Keine Farbskala verwenden, um den Fehler zu vermeiden
        st.pyplot(plt)

    # Positionale Kodierung
    st.title("📏 Positionale Kodierung Visualisierung")

    def positional_encoding_simple(token_index, max_len=10):
        return np.array([(token_index / max_len), (token_index / max_len) ** 2])

    # Berechnung der Positionale Kodierungen
    positional_encodings = [positional_encoding_simple(i) for i in range(len(tokens))]

    # Interaktives Balkendiagramm
    for i, (token, encoding) in enumerate(zip(tokens, positional_encodings)):
        st.subheader(f"Token: {token}")
        st.bar_chart(encoding)

    # Möglichkeit, die Reihenfolge der Tokens zu ändern
    st.subheader("Ändern Sie die Reihenfolge der Wörter:")
    tokens_shuffled = st.multiselect('Neue Reihenfolge:', tokens, default=tokens)

    if tokens_shuffled:
        # Neuberechnung der Positionale Kodierung für die neue Reihenfolge
        positional_encodings_shuffled = [positional_encoding_simple(i) for i in range(len(tokens_shuffled))]

        for i, (token, encoding) in enumerate(zip(tokens_shuffled, positional_encodings_shuffled)):
            st.subheader(f"Token: {token}")
            st.bar_chart(encoding)
