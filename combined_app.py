import streamlit as st
import gensim.downloader as api
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objs as go
from transformers import AutoTokenizer
import random

hf_token = st.secrets["hf_token"]

# Set page config
st.set_page_config(page_title="Word Embeddings & Sentence Tokenizer", layout="wide")

# General styling for consistency
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
st.title("🔍 Word Embeddings Visualizations")

# Load the GloVe model only once using caching
@st.cache_resource
def load_model():
    return api.load('glove-wiki-gigaword-50')

# Load the model
model = load_model()

# Word groups
animal_words = ["dog", "cat", "lion", "elephant", "bird", "fish", "horse", "tiger", "whale", "bear"]
fruit_words = ["apple", "banana", "cherry", "grape", "orange", "pear", "peach", "plum", "kiwi", "mango"]
color_words = ["red", "blue", "green", "yellow", "purple", "pink", "orange", "black", "white", "brown"]
emotion_words = ["happy", "sad", "angry", "excited", "nervous", "fear", "joy", "love", "hate", "surprise"]

# Combine all words
all_words = animal_words + fruit_words + color_words + emotion_words

# Get embeddings for all the words
embeddings = np.array([model[word] for word in all_words])

# 2D PCA
pca_2d = PCA(n_components=2)
reduced_embeddings_2d = pca_2d.fit_transform(embeddings)

# 3D PCA
pca_3d = PCA(n_components=3)
reduced_embeddings_3d = pca_3d.fit_transform(embeddings)

# Create the 2D scatter plot
fig_2d = go.Figure()

# Add points for each word group
for i, words in enumerate([animal_words, fruit_words, color_words, emotion_words]):
    fig_2d.add_trace(go.Scatter(
        x=reduced_embeddings_2d[i*10:(i+1)*10, 0],
        y=reduced_embeddings_2d[i*10:(i+1)*10, 1],
        mode='markers+text',
        text=words,
        marker=dict(
            size=12,
            color=['green', 'orange', 'blue', 'red'][i]
        ),
        textposition="top center",
        name=f'{["Animal", "Fruit", "Color", "Emotion"][i]}-related words'
    ))

# Update 2D layout
fig_2d.update_layout(
    xaxis_title='PCA 1',
    yaxis_title='PCA 2',
    height=600,
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

# Create the 3D scatter plot
fig_3d = go.Figure()

# Add points for each word group
for i, words in enumerate([animal_words, fruit_words, color_words, emotion_words]):
    fig_3d.add_trace(go.Scatter3d(
        x=reduced_embeddings_3d[i*10:(i+1)*10, 0],
        y=reduced_embeddings_3d[i*10:(i+1)*10, 1],
        z=reduced_embeddings_3d[i*10:(i+1)*10, 2],
        mode='markers+text',
        text=words,
        marker=dict(
            size=8,
            color=['green', 'orange', 'blue', 'red'][i]
        ),
        textposition="top center",
        name=f'{["Animal", "Fruit", "Color", "Emotion"][i]}-related words'
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
st.subheader("2D and 3D Word Embeddings Visualizations")

# Create two columns
col1, col2 = st.columns(2)

# Display charts side by side
with col1:
    st.write("#### 2D Visualization")
    st.plotly_chart(fig_2d, use_container_width=True, config={'displayModeBar': True, 'scrollZoom': True})

with col2:
    st.write("#### 3D Visualization")
    st.plotly_chart(fig_3d, use_container_width=True, config={'displayModeBar': True, 'scrollZoom': True})

# Add a larger padding for clearance
st.markdown("<div style='height: 150px;'></div>", unsafe_allow_html=True)

# Section 2: Sentence Tokenizer
st.title("📝 Sentence Tokenizer")

# Define a set of subtle, semi-transparent colors
COLORS = [
    "rgba(255, 99, 71, 0.3)",   # Tomato
    "rgba(255, 165, 0, 0.3)",   # Orange
    "rgba(255, 215, 0, 0.3)",   # Gold
    "rgba(154, 205, 50, 0.3)",  # Yellow Green
    "rgba(0, 255, 127, 0.3)",   # Spring Green
    "rgba(100, 149, 237, 0.3)", # Cornflower Blue
    "rgba(138, 43, 226, 0.3)",  # Blue Violet
    "rgba(255, 192, 203, 0.3)", # Pink
]

# Function to get a random color, avoiding consecutive repeats
def get_random_color(previous_color=None):
    available_colors = [c for c in COLORS if c != previous_color]
    return random.choice(available_colors)

# Initialize tokenizer
@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", use_auth_token=hf_token)

tokenizer = load_tokenizer()

# Create three columns
col1, col2, col3 = st.columns([2, 1, 1])

# Placeholder text
placeholder_text = "Ein verlassener Garten verwilderte. Ein Junge begann, ihn zu pflegen. Blumen wuchsen bald überall."

# User input in the first column
with col1:
    user_input = st.text_area("Enter a sentence or query:", value=placeholder_text, height=100)

if user_input:
    # Tokenize the input
    tokens = tokenizer.tokenize(user_input)

    # Display tokenized result
    st.subheader("Tokenized Result:")

    # Create a container for tokens with line breaks
    tokens_html = '<div style="line-height: 1.6; text-align: left; word-break: break-word;">'
    previous_color = None
    new_sentence = True
    for token in tokens:
        color = get_random_color(previous_color)
        cleaned_token = token.replace('▁', ' ').strip()  # Replace '▁' with space and strip
        if cleaned_token:  # Only add non-empty tokens
            if new_sentence:
                tokens_html += '<div style="margin-bottom: 6px;">'
            tokens_html += f'<span style="background-color:{color}; padding:3px 8px; border-radius:4px; margin-right:6px; margin-bottom:6px; display:inline-block; font-size:1.05em;">{cleaned_token}</span>'
            if cleaned_token.endswith('.'):
                tokens_html += '</div><div style="margin-bottom: 18px;"></div>'  # Increased space after a period
                new_sentence = True
            else:
                new_sentence = False
        previous_color = color
    if not new_sentence:
        tokens_html += '</div>'
    tokens_html += '</div>'

    st.markdown(tokens_html, unsafe_allow_html=True)

    # Display token information
    st.subheader("Token Information:")
    st.write(f"Number of tokens: {len(tokens)}")

    # Show token IDs
    token_ids = tokenizer.encode(user_input, add_special_tokens=False)
    st.write("Token IDs Hallo Jonas:", token_ids)
