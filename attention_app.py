import streamlit as st
import numpy as np
import plotly.graph_objs as go

# Funktion, um zufällige Aufmerksamkeitswerte zu generieren
def generate_attention_values(num_words):
    np.random.seed(42)  # Für Reproduzierbarkeit
    return np.random.rand(num_words, num_words)

# Streamlit App Layout
st.title("Self-Attention: Das Spotlight der Aufmerksamkeit")

st.markdown("""
Stellen Sie sich vor, die Aufmerksamkeit in einem Transformermodell ist wie ein Spotlight, das auf verschiedene Wörter im Satz gerichtet wird. Je heller das Licht, desto mehr Aufmerksamkeit wird diesem Wort geschenkt.
""")

# Benutzereingabe für den Satz
user_sentence = st.text_input("Geben Sie einen Satz ein:", value="Die Katze saß auf der Matte")
words = user_sentence.split()

# Generierung zufälliger Aufmerksamkeitswerte
attention_values = generate_attention_values(len(words))

# Auswahl des Zielwortes
target_word_index = st.slider("Wählen Sie das Wort aus, auf das das Spotlight gerichtet werden soll:", 0, len(words) - 1, 0)

# Erzeugung der Spotlight-Darstellung
spotlight_intensity = attention_values[target_word_index]

# Visualisierung
fig = go.Figure()

# Visualisierung der Spotlight-Intensität für jedes Wort
for i, word in enumerate(words):
    fig.add_trace(go.Scatter(
        x=[i],
        y=[1],
        text=[word],
        mode='markers+text',
        textposition="bottom center",
        marker=dict(size=20, color=f'rgba(255, 255, 0, {spotlight_intensity[i]})'),
        showlegend=False
    ))

fig.update_layout(
    xaxis=dict(tickvals=list(range(len(words))), ticktext=words, title="Wörter im Satz"),
    yaxis=dict(showticklabels=False),
    height=300,
    title=f"Spotlight-Intensität auf das Wort: '{words[target_word_index]}'"
)

st.plotly_chart(fig)

st.markdown(f"""
### Erklärung:
In dieser Visualisierung haben Sie das Wort '{words[target_word_index]}' ausgewählt. 
Das Spotlight zeigt, wie viel Aufmerksamkeit dieses Wort auf die anderen Wörter im Satz richtet. 
Je heller ein Wort dargestellt wird, desto mehr Aufmerksamkeit erhält es von '{words[target_word_index]}'.
""")
