import streamlit as st
from transformers import AutoTokenizer
import random

# Set page config
st.set_page_config(page_title="Sentence Tokenizer", layout="wide")

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
    return AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

tokenizer = load_tokenizer()

# App title
st.title("📝 Sentence Tokenizer")

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
    st.write("Token IDs:", token_ids)

# Add some styling
st.markdown("""
<style>
    .stApp {
        max-width: 100%;
        padding: 0 2rem;
    }
    .st-emotion-cache-16idsys p {
        font-size: 1.2rem;
    }
    .stTextArea textarea {
        font-size: 1rem;
    }
    /* Target the helper text */
    .stTextArea .st-emotion-cache-16txtl3 {
        font-size: 0.8rem;
        opacity: 0.7;
    }
</style>
""", unsafe_allow_html=True)