import json
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from openai import OpenAI
import os
from token_tracker import track_token_usage, calculate_total_cost  # Importiere Token Tracker

# Lade Umgebungsvariablen aus der .env-Datei
load_dotenv()

# Setze Umgebungsvariablen
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

client = OpenAI()

model = "gpt-4o-2024-08-06"

# Initialisiere Embeddings und Vektor-Datenbank für RAG
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = Chroma(persist_directory="./vectordb/vertrag", embedding_function=embeddings)

def retrieve_context(question, k=1):
    """Rufe relevanten Kontext aus dem Chroma-Vektor-Speicher basierend auf der Frage ab."""
    # Führe eine Ähnlichkeitssuche durch, um relevante Dokumente zu erhalten
    results = vectorstore.similarity_search(question, k=k)
    
    # Rückgabe der relevantesten Chunks des Kontextes
    context = ""
    for res in results:
        context += f"{res.page_content}\n\n{res.metadata}\n\n"
        print(f"Abgerufener Chunk: \n{res.page_content} \n[{res.metadata}]")
    
    return context

def build_prompt(question, context):
    return f"""
    FRAGE: {question}

    KONTEXT:
    {context}
    """

def call_llm(prompt):
    """Rufe die OpenAI API mit dem erstellten Prompt auf."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": """Beantworte diese FRAGE nur unter Verwendung des bereitgestellten KONTEXT. 
                                                Strukturiere die Antwort in einer klaren und geordneten Weise."""},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    # Verfolgung der Token-Nutzung
    track_token_usage(response.usage, model)  # Übergebe die Nutzungsdaten und das Modell

    # Greife direkt auf den Inhalt der ersten Antwortnachricht zu
    return response.choices[0].message.content

def process_rag_query_from_json(json_file_path):
    # Lade die JSON-Datei
    with open(json_file_path, 'r') as f:
        discrepancies = json.load(f)

    # Extrahiere die 'difference_in_days' aus den Datumsabweichungen
    if discrepancies['date_discrepancies']:
        difference_in_days = discrepancies['date_discrepancies'][0]['difference_in_days']
    else:
        # Beende den Prozess, wenn keine Differenz gefunden wurde, damit keine RAG-Abfrage gesendet wird
        print("Keine Abweichungen in der Lieferzeit gefunden, RAG-Abfrage wird nicht gesendet.")
        return None

    # Erstelle den Fragenstring mit der extrahierten 'difference_in_days'

    question = f"""Die Lieferzeit der Ware betrug: {difference_in_days} Tage. 
                    Stimmt das mit der vertraglichen Vereinbarung überein?"""

    print(f"Frage, die an RAG gesendet wird: {question}")

    # Rufe relevanten Kontext aus der Vektordatenbank ab
    context = retrieve_context(question)

    # Erstelle den Prompt für das LLM
    prompt = build_prompt(question, context)

    # Rufe das LLM mit dem erstellten Prompt auf und erhalte die Antwort
    response = call_llm(prompt)

    return response

# Berechnung der Gesamtkosten für die API Calls
def calculate_rag_total_cost():
    return calculate_total_cost()
