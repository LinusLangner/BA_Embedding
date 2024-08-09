import streamlit as st
import gensim.downloader as api
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objs as go

# Set page to wide mode
st.set_page_config(layout="wide")

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
st.write("### 2D and 3D Word Embeddings Visualizations")

# Create two columns
col1, col2 = st.columns(2)

# Display charts side by side
with col1:
    st.write("#### 2D Visualization")
    st.plotly_chart(fig_2d, use_container_width=True, config={'displayModeBar': True, 'scrollZoom': True})

with col2:
    st.write("#### 3D Visualization")
    st.plotly_chart(fig_3d, use_container_width=True, config={'displayModeBar': True, 'scrollZoom': True})

# Instructions for interaction
st.write("""
- Linus Langner
- 2557735
- HAW Hamburg
""")
