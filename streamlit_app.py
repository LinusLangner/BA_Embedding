import streamlit as st
import gensim.downloader as api
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objs as go
import tiktoken
import random
import openai
import os

hf_token = st.secrets["hf_token"]
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Seiteneinstellungen
st.set_page_config(page_title="Theorie - Methodik", page_icon="📈", layout="wide")

# Umfassende Einführung am Anfang der App
st.markdown("""
<div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 30px; border-left: 5px solid #0066cc;">
    <h2 style="color: #0066cc; margin-top: 0;">Bachelorarbeit: Automatisierte Dokumentenverarbeitung in der Bekleidungsindustrie</h2>
    <h3 style="color: #333;">Theoretischer Hintergrund und Methodik</h3>
    <p style="font-size: 16px; line-height: 1.6;">
        <strong>Titel der Bachelorarbeit:</strong> Innovationen durch Künstliche Intelligenz: Automatisierte Dokumentenverarbeitung in der Bekleidungsindustrie
    </p>
    <p style="font-size: 16px; line-height: 1.6;">
        Diese Anwendung demonstriert den theoretischen Hintergrund und die Methodik der Forschung zur automatisierten Dokumentenverarbeitung unter Einsatz künstlicher Intelligenz in der Bekleidungsindustrie. Sie bietet Einblicke in die zugrunde liegenden Konzepte und Forschungsansätze.
    </p>
    <p style="font-size: 16px; line-height: 1.6;">
        <strong>Student:</strong> Linus Langner<br>
        <strong>Semester:</strong> 9. Semester BTM SS24<br>
        <strong>Matrikelnummer:</strong> 2557735
    </p>
    <p style="font-size: 14px; font-style: italic;">
        Entwickelt im Rahmen der Bachelorarbeit an der HAW Hamburg - Fakultät DMI - Department Design
    </p>
</div>
""", unsafe_allow_html=True)

# Allgemeine Stilrichtlinien für Konsistenz
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

# Abschnitt 1: Wort-Embeddings-Visualisierungen
st.title("🔍 Wort-Embeddings Visualisierungen")

# GloVe-Modell nur einmal laden und zwischenspeichern
@st.cache_resource
def load_model():
    return api.load('glove-wiki-gigaword-50')

# Modell laden
model = load_model()

# Wortgruppen
tier_worte = ["dog", "cat", "lion", "elephant", "bird", "fish", "horse", "tiger", "whale", "bear"]
obst_worte = ["apple", "banana", "cherry", "grape", "orange", "pear", "peach", "plum", "peanut", "mango"]
farben_worte = ["red", "blue", "green", "yellow", "purple", "pink", "black", "white", "brown"]
emotions_worte = ["happy", "sad", "angry", "excited", "nervous", "fear", "joy", "love", "hate", "surprise"]

# Benutzereingabe für benutzerdefinierte Wörter
st.subheader("Fügen Sie Ihre eigenen Wörter hinzu")
st.write("Bitte beachten Sie, dass die Wörter auf Englisch eingegeben werden sollten.")
user_words = st.text_input("Geben Sie Wörter ein, getrennt durch ein Komma und Leertaste: (Wort1, Wort2, Wort3, usw.)", "")
user_words = [word.strip().lower() for word in user_words.split(',') if word.strip()]

# Alle Wörter kombinieren
alle_worte = tier_worte + obst_worte + farben_worte + emotions_worte + user_words

# Einbettungen für alle Wörter erhalten, nur wenn das Wort im Modell vorhanden ist
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
    # 2D und 3D PCA
    pca_2d = PCA(n_components=2)
    reduzierte_embeddings_2d = pca_2d.fit_transform(embeddings)

    pca_3d = PCA(n_components=3)
    reduzierte_embeddings_3d = pca_3d.fit_transform(embeddings)

    # Markierungsformen und -größen für verschiedene Gruppen definieren
    shapes_2d = ['circle', 'square', 'diamond', 'triangle-up', 'cross']
    sizes_3d = [8, 10, 12, 14, 16]  # Simulieren verschiedener "Formen" mit Größen in 3D

    # 2D-Streudiagramm
    fig_2d = go.Figure()

    # Punkte für jede Wortgruppe hinzufügen
    colors = ['green', 'orange', 'blue', 'red', '#00FFFF']  # Cyan für Benutzereingaben
    group_names = ["Tier", "Obst", "Farb", "Emotions", "Benutzer Eingaben"]

    for i, words in enumerate([tier_worte, obst_worte, farben_worte, emotions_worte, user_words]):
        valid_indices = [j for j, word in enumerate(valid_words) if word in words]
        if valid_indices:
            # Benutzer Eingaben ohne "-bezogene Wörter"
            group_name = group_names[i] if i == 4 else f'{group_names[i]}-bezogene Wörter'
            fig_2d.add_trace(go.Scatter(
                x=reduzierte_embeddings_2d[valid_indices, 0],
                y=reduzierte_embeddings_2d[valid_indices, 1],
                mode='markers+text',
                text=[valid_words[j] for j in valid_indices],
                marker=dict(
                    size=12,
                    color=colors[i],
                    symbol=shapes_2d[i]
                ),
                textposition="top center",
                name=group_name
            ))

    # 2D-Layout mit anfänglichem Herauszoomen aktualisieren
    x_min, x_max = reduzierte_embeddings_2d[:, 0].min(), reduzierte_embeddings_2d[:, 0].max()
    y_min, y_max = reduzierte_embeddings_2d[:, 1].min(), reduzierte_embeddings_2d[:, 1].max()

    fig_2d.update_layout(
        xaxis_title='PCA 1',
        yaxis_title='PCA 2',
        height=600,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(range=[x_min - 1, x_max + 1]),
        yaxis=dict(range=[y_min - 1, y_max + 1]),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    # 3D-Streudiagramm
    fig_3d = go.Figure()

    # Punkte für jede Wortgruppe hinzufügen
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
                    size=sizes_3d[i],
                    color=colors[i],
                ),
                textposition="top center",
                name=f'{group_names[i]}-bezogene Wörter' if i < 4 else group_names[i]
            ))

    # 3D-Layout aktualisieren
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
        st.write("Hinweis: Ein Doppelklick in die Anwendung setzt die Ansicht zurück.")
        st.plotly_chart(fig_2d, use_container_width=True, config={'displayModeBar': True, 'scrollZoom': True})

    with col2:
        st.write("#### 3D Visualisierung")
        st.write("Hinweis: Durch Klicken und Ziehen können Sie die Ansicht drehen.")
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

# Funktion zur Tokenisierung unter Verwendung von tiktoken
@st.cache_resource
def load_tokenizer(encoding_name="cl100k_base"):
    return tiktoken.get_encoding(encoding_name)

def tokens_from_string(string: str, encoding_name: str):
    """Gibt die Tokens in einem Textstring zurück."""
    encoding = load_tokenizer(encoding_name)
    token_ids = encoding.encode(string)
    tokens = [encoding.decode_single_token_bytes(token_id).decode('utf-8') for token_id in token_ids]
    return tokens, token_ids

# Drei Spalten erstellen
col1, col2, col3 = st.columns([2, 1, 1])

# Platzhaltertext
platzhalter_text = "Die Katze springt auf den Tisch."

# Benutzereingabe in der ersten Spalte
with col1:
    user_input = st.text_area("Geben Sie einen Satz oder eine Abfrage ein:", value=platzhalter_text, height=100)

if user_input:
    # Eingabe tokenisieren
    tokens, token_ids = tokens_from_string(user_input, "cl100k_base")

    # Tokenisiertes Ergebnis anzeigen
    st.subheader("Tokenisiertes Ergebnis:")

    # Container für Tokens mit Zeilenumbrüchen erstellen
    tokens_html = '<div style="line-height: 1.6; text-align: left; word-break: break-word;">'
    previous_color = None
    new_sentence = True
    for token in tokens:
        color = get_random_color(previous_color)
        tokens_html += f'<span style="background-color:{color}; padding:3px 8px; border-radius:4px; margin-right:6px; margin-bottom:6px; display:inline-block; font-size:1.05em;">{token}</span>'
        if token.endswith('.'):
            tokens_html += '</div><div style="margin-bottom: 18px;"></div>'  # Größerer Abstand nach einem Punkt
            new_sentence = True
        else:
            new_sentence = False
        previous_color = color
    if not new_sentence:
        tokens_html += '</div>'
    tokens_html += '</div>'

    st.markdown(tokens_html, unsafe_allow_html=True)

    st.write(f"Anzahl der Tokens: {len(tokens)}")


# Größeren Abstand für klare Trennung hinzufügen
st.markdown("<div style='height: 150px;'></div>", unsafe_allow_html=True)

# Abschnitt für API-Vergleich
st.title("🔍 API Vergleich (Temperatur 0 vs 0.7)")
st.write("Es bietet sich an, die jeweiligen Beispiele mehrmals auszuführen, um die Unterschiede in den Antworten zu sehen.")

# Initialisiere session_state
if 'api_input' not in st.session_state:
    st.session_state.api_input = ""
if 'run_api' not in st.session_state:
    st.session_state.run_api = False

# Funktion zum Setzen der Eingabe und Auslösen des API-Aufrufs
def set_input_and_run(input_text):
    st.session_state.api_input = input_text
    st.session_state.run_api = True

# Custom CSS für die Buttons
st.markdown("""
<style>
    .stButton > button {
        background-color: #0066cc;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        margin-bottom: 10px;
        width: 200px;  /* Feste Breite für alle Buttons */
        text-align: left;
    }
    .stButton > button:hover {
        background-color: #0056b3;
    }
    .button-container {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
    }
</style>
""", unsafe_allow_html=True)

# Beispiele als Buttons übereinander und linksbündig
st.subheader("Beispiele")
with st.container():
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    if st.button("Zufallszahl"):
        set_input_and_run("Nenne eine zufällige Zahl zwischen 0 und 100.")
    if st.button("Witz"):
        set_input_and_run("Erzähle mir einen Witz.")
    st.markdown('</div>', unsafe_allow_html=True)

# Freie Benutzereingabe
user_input = st.text_input("Geben Sie Ihre eigene Abfrage ein:", value=st.session_state.api_input)
if user_input != st.session_state.api_input:
    st.session_state.api_input = user_input
    st.session_state.run_api = True
    

# Funktion für API-Aufruf
def call_openai_api(api_input, temp):
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Du bist ein hilfreicher Assistent."}, 
            {"role": "user", "content": api_input}
        ],
        temperature=temp
    )
    return response.choices[0].message.content

# API-Aufrufe durchführen, wenn eine Eingabe vorhanden ist und run_api True ist
if st.session_state.api_input and st.session_state.run_api:
    with st.spinner('API-Anfragen werden verarbeitet...'):
        # API mit Temperatur 0 und 0.7 aufrufen
        st.write(f"Calling API with temperature 0")
        response_temp_0 = call_openai_api(st.session_state.api_input, 0)
        st.write(f"Calling API with temperature 0.7")
        response_temp_07 = call_openai_api(st.session_state.api_input, 0.7)

        # Beide Antworten nebeneinander mit Rahmen und Markdown anzeigen
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Antwort (Temperatur 0)")
            st.markdown(
                f"<div style='border: 2px solid #4CAF50; padding: 10px; border-radius: 10px;'>{response_temp_0}</div>", 
                unsafe_allow_html=True
            )

        with col2:
            st.subheader("Antwort (Temperatur 0.7)")
            st.markdown(
                f"<div style='border: 2px solid #FF9800; padding: 10px; border-radius: 10px;'>{response_temp_07}</div>", 
                unsafe_allow_html=True
            )

    # Zurücksetzen von run_api nach der Ausführung
    st.session_state.run_api = False