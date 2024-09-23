import streamlit as st
import os
import json
import time
import fitz  # PyMuPDF
from pydantic import BaseModel
from openai import OpenAI
from typing import List, Dict, Any
from datetime import datetime
import chromadb
from dotenv import load_dotenv
import chromadb.utils.embedding_functions as embedding_functions

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OpenAI API key not found. Please set it in the .env file.")

# Laden des OpenAI API-Schlüssels aus Streamlit secrets
# openai_api_key = st.secrets["OPENAI_API_KEY"]

# OpenAI-Client einrichten
client = OpenAI(api_key=openai_api_key)

# Modelle definieren
model = "gpt-4o-2024-08-06"
model_mini = "gpt-4o-mini"

# Einrichten des Chroma HTTP-Clients
chroma_client = chromadb.HttpClient(
    host='16.171.0.22',  # Ersetzen Sie dies durch die öffentliche IP-Adresse Ihrer AWS-Instanz
    port=8000
)

# Verwendung von OpenAIEmbeddingFunction aus Chroma utils
embedding_function = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_api_key,
    model_name="text-embedding-3-large"
)

# Sammlung abrufen oder erstellen
collection = chroma_client.get_or_create_collection(
    name="bachelor-vertrag",
    embedding_function=embedding_function
)

# DocumentExtraction-Klasse definieren
class DocumentExtraction(BaseModel):
    order_number: str
    article_number: List[str]
    description: List[str]
    amount: List[int]
    prices: List[float]
    total: float
    date: int

# Token-Tracking und Kostenberechnung
token_usage = {
    "gpt-4o-2024-08-06": {"input_tokens": 0, "output_tokens": 0},
    "gpt-4o-mini": {"input_tokens": 0, "output_tokens": 0}
}

def track_token_usage(usage, model_type):
    global token_usage
    token_usage[model_type]["input_tokens"] += usage.prompt_tokens
    token_usage[model_type]["output_tokens"] += usage.completion_tokens

def calculate_total_cost():
    total_cost = 0.0
    for model_type, usage in token_usage.items():
        if model_type == 'gpt-4o-2024-08-06':
            input_cost_per_million = 2.500
            output_cost_per_million = 10.000
        elif model_type == 'gpt-4o-mini':
            input_cost_per_million = 0.150
            output_cost_per_million = 0.600
        input_cost = (usage["input_tokens"] / 1_000_000) * input_cost_per_million
        output_cost = (usage["output_tokens"] / 1_000_000) * output_cost_per_million
        total_cost += input_cost + output_cost
    return round(total_cost, 4)

# Funktionen zur Extraktion strukturierter Daten
def create_pdf_path_for_order(order_number):
    pdf_path = f"documents/Bestellaufträge/{order_number}.pdf"
    if os.path.exists(pdf_path):
        st.success(f"✅ Bestellung {order_number} anhand der extrahierten order_number im Dateisystem gefunden.")
    else:
        st.error(f"❌ Bestellnummer {order_number} nicht gefunden.")
    return pdf_path

def extract_text_from_pdf(pdf_path):
    with st.spinner(f"Extrahiere Text aus PDF: {pdf_path}"):
        pdf_document = fitz.open(pdf_path)
        text = ""
        for page in pdf_document:
            text += page.get_text("text")
        pdf_document.close()
    st.success("✅ Text erfolgreich extrahiert.")
    return text

def extract_structured_data(user_input):
    with st.spinner("Extrahiere strukturierte Daten..."):
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": """Sie sind ein Experte für die Extraktion strukturierter Daten."""},
                {"role": "user", "content": user_input}
            ],
            response_format=DocumentExtraction,
            temperature=0.0
        )
    track_token_usage(completion.usage, model)
    structured_data = completion.choices[0].message.parsed
    return structured_data

def process_order_and_invoice(invoice_pdf_filename):
    invoice_path = f"documents/Lieferantenrechnungen/{invoice_pdf_filename}"
    invoice_content = extract_text_from_pdf(invoice_path)
    invoice_structured_data = extract_structured_data(invoice_content)
    st.success("✅ Strukturierte Rechnungsdaten extrahiert:")
    st.json(invoice_structured_data.dict())

    order_number = invoice_structured_data.order_number
    order_path = create_pdf_path_for_order(order_number)
    order_content = extract_text_from_pdf(order_path)
    order_structured_data = extract_structured_data(order_content)
    st.success("✅ Strukturierte Bestelldaten extrahiert:")
    st.json(order_structured_data.dict())

    return order_structured_data, invoice_structured_data

# Vergleichsfunktionen
def compare_article_numbers_with_gpt(order_article_numbers: List[str], invoice_article_numbers: List[str]) -> List[Dict[str, Any]]:
    discrepancies = []
    st.subheader("Vergleich der Artikelnummern")
    for invoice_article in invoice_article_numbers:
        with st.spinner(f"Prüfe Rechnungsartikelnummer: {invoice_article}"):
            response = client.chat.completions.create(
                model=model_mini,
                messages=[
                    {"role": "system", "content": """Sie sind ein Experte im Vergleichen von Artikelnummern."""},
                    {"role": "user", "content": f"""Rechnungsartikelnummer: {invoice_article}. 
                                                    Stimmt sie mit einer der folgenden Bestellartikelnummern überein: {order_article_numbers}?"""}
                ],
                temperature=0.0
            )

        track_token_usage(response.usage, model_mini)
        gpt_response = response.choices[0].message.content
        
        if gpt_response.lower().startswith("nein"):
            discrepancies.append({
                "invoice_article_number": invoice_article,
                "order_article_numbers": order_article_numbers,
                "note": """Rechnungsartikelnummer stimmt mit keiner Bestellartikelnummer überein."""
            })
            st.warning(f"⚠️ Abweichung gefunden: {invoice_article} nicht in Bestellung")
        else:
            st.success(f"✅ {invoice_article} in Bestellung gefunden")
            
    return discrepancies

def compare_amounts(order_amounts: List[int], invoice_amounts: List[int]) -> List[Dict[str, Any]]:
    discrepancies = []
    st.subheader("Vergleich der Mengen")
    for i, order_amount in enumerate(order_amounts):
        invoice_amount = invoice_amounts[i] if i < len(invoice_amounts) else None
        if order_amount != invoice_amount:
            discrepancies.append({
                "article_number": i + 1,
                "order_amount": order_amount,
                "invoice_amount": invoice_amount,
                "difference": order_amount - invoice_amount if invoice_amount is not None else "Nicht angegeben"
            })
            st.warning(f"⚠️ Mengenabweichung bei Artikel {i+1}: Bestellung {order_amount}, Rechnung {invoice_amount}")
        else:
            st.success(f"✅ Mengen stimmen überein für Artikel {i+1}: {order_amount}")
    return discrepancies

def compare_prices(order_prices: List[float], invoice_prices: List[float]) -> List[Dict[str, Any]]:
    discrepancies = []
    st.subheader("Vergleich der Preise")
    for i, order_price in enumerate(order_prices):
        invoice_price = invoice_prices[i] if i < len(invoice_prices) else None
        if order_price != invoice_price:
            discrepancies.append({
                "article_number": i + 1,
                "order_price": order_price,
                "invoice_price": invoice_price,
                "difference": order_price - invoice_price if invoice_price is not None else "Nicht angegeben"
            })
            st.warning(f"⚠️ Preisabweichung bei Artikel {i+1}: Bestellung {order_price}€, Rechnung {invoice_price}€")
        else:
            st.success(f"✅ Preise stimmen überein für Artikel {i+1}: {order_price}€")
    return discrepancies

def compare_total(order_total: float, invoice_total: float) -> List[Dict[str, Any]]:
    st.subheader("Vergleich der Gesamtbeträge")
    discrepancies = []
    if order_total != invoice_total:
        discrepancies.append({
            "order_total": order_total,
            "invoice_total": invoice_total,
            "difference": order_total - invoice_total
        })
        st.warning(f"⚠️ Gesamtbetrag weicht ab: Bestellung {order_total}€, Rechnung {invoice_total}€")
    else:
        st.success(f"✅ Gesamtbeträge stimmen überein: {order_total}€")
    return discrepancies

def compare_descriptions_with_gpt(order_descriptions: List[str], invoice_descriptions: List[str]) -> List[Dict[str, Any]]:
    discrepancies = []
    st.subheader("Vergleich der Beschreibungen")
    for i, order_desc in enumerate(order_descriptions):
        invoice_desc = invoice_descriptions[i] if i < len(invoice_descriptions) else None
        with st.spinner(f"Vergleiche Beschreibungen für Artikel {i+1}"):
            response = client.chat.completions.create(
                model=model_mini,
                messages=[
                    {"role": "system", "content": """Sie sind ein Experte im Vergleichen von Produktbeschreibungen."""},
                    {"role": "user", "content": f"Beschreiben die folgenden Beschreibungen das gleiche Produkt? "
                                                f"Bestellbeschreibung: {order_desc} "
                                                f"Rechnungsbeschreibung: {invoice_desc}"}
                ],
                temperature=0.0
            )

        track_token_usage(response.usage, model_mini)
        
        gpt_response = response.choices[0].message.content.lower()
        if gpt_response.lower().startswith("nein"):
            discrepancies.append({
                "article_number": i + 1,
                "order_description": order_desc,
                "invoice_description": invoice_desc,
                "note": "Beschreibungen sind unterschiedlich."
            })
            st.warning(f"⚠️ Beschreibungsabweichung bei Artikel {i+1}")
        else:
            st.success(f"✅ Beschreibungen stimmen überein für Artikel {i+1}")

    return discrepancies

def compare_dates(order_date: int, invoice_date: int) -> List[Dict[str, Any]]:
    st.subheader("Vergleich der Daten")
    order_date_obj = datetime.strptime(str(order_date), "%Y%m%d")
    invoice_date_obj = datetime.strptime(str(invoice_date), "%Y%m%d")
    difference_in_days = (invoice_date_obj - order_date_obj).days

    discrepancies = [{
        "order_date": order_date,
        "invoice_date": invoice_date,
        "difference_in_days": difference_in_days
    }]
    st.info(f"📅 Zeitunterschied zwischen Bestellung und Rechnung: {difference_in_days} Tage")
    return discrepancies

def comparer(order_structured_data: DocumentExtraction, invoice_structured_data: DocumentExtraction) -> Dict[str, Any]:
    st.write("Starte den Vergleich von Bestell- und Rechnungsdaten...")
    discrepancies = {
        "article_number_discrepancies": compare_article_numbers_with_gpt(order_structured_data.article_number, invoice_structured_data.article_number),
        "quantity_discrepancies": compare_amounts(order_structured_data.amount, invoice_structured_data.amount),
        "price_discrepancies": compare_prices(order_structured_data.prices, invoice_structured_data.prices),
        "total_discrepancies": compare_total(float(order_structured_data.total), float(invoice_structured_data.total)),
        "description_discrepancies": compare_descriptions_with_gpt(order_structured_data.description, invoice_structured_data.description),
        "date_discrepancies": compare_dates(order_structured_data.date, invoice_structured_data.date)
    }
    st.success("✅ Vergleich abgeschlossen.")
    return discrepancies

# RAG-Funktionen
def retrieve_context(question, k=1):
    with st.spinner("Suche relevante Vertragsklauseln..."):
        results = collection.query(
            query_texts=[question],
            n_results=k,
            include=["documents", "metadatas"]
        )
    context = ""
    for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
        context += f"{doc}\n\n{metadata}\n\n"
        st.info(f"📄 Gefundene relevante Klausel:  \n{doc}")
        st.info(f"📄 Ursprung der Klausel:  \n{metadata}")
    return context

def build_prompt(question, context):
    return f"""
    FRAGE: {question}

    KONTEXT:
    {context}
    """

def call_llm(prompt):
    with st.spinner("Analysiere Vertragsklauseln..."):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": """Beantworten Sie die FRAGE nur mit dem bereitgestellten KONTEXT."""},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

    track_token_usage(response.usage, model)

    return response.choices[0].message.content

def process_rag_query_from_json(json_file_path):
    with open(json_file_path, 'r') as f:
        discrepancies = json.load(f)

    if discrepancies['date_discrepancies']:
        difference_in_days = discrepancies['date_discrepancies'][0]['difference_in_days']
    else:
        st.info("ℹ️ Keine Lieferzeitabweichungen gefunden, überspringe RAG-Abfrage.")
        return None

    question = f"Die Lieferzeit betrug {difference_in_days} Tage. Entspricht dies der vertraglichen Vereinbarung?"
    st.write(f"🔍 Analysiere folgende Frage: {question}")

    context = retrieve_context(question)
    prompt = build_prompt(question, context)

    response = call_llm(prompt)

    return response

def calculate_rag_total_cost():
    return calculate_total_cost()

def main(invoice_pdf_filename):
    start_time = time.time()

    st.header("📊 Datenextraktion und -verarbeitung")
    with st.spinner("Verarbeite Bestellung und Rechnung..."):
        order_structured_data, invoice_structured_data = process_order_and_invoice(invoice_pdf_filename)

    st.header("🔍 Datenvergleich")
    discrepancies = comparer(order_structured_data, invoice_structured_data)
    
    st.subheader("Zusammenfassung der Abweichungen:")
    if any(discrepancies.values()):
        for category, issues in discrepancies.items():
            if issues:
                st.warning(f"⚠️ {category.replace('_', ' ').capitalize()}: {len(issues)} Abweichung(en) gefunden")
    else:
        st.success("✅ Keine Abweichungen gefunden!")

    invoice_base_name = os.path.splitext(invoice_pdf_filename)[0]
    order_number = invoice_structured_data.order_number
    json_filename = f"{invoice_base_name}_{order_number}.json"

    with open(f"documents/Diskrepanzen/{json_filename}", "w") as f:
        json.dump(discrepancies, f, indent=4)

    st.success(f"✅ Abweichungen gespeichert in {json_filename}.")

    st.header("🤖 RAG-Prozess")
    st.write("Überprüfe Lieferzeit anhand der Vertragsklauseln...")
    
    total_cost_before_rag = calculate_total_cost()
    
    response = process_rag_query_from_json(f"documents/Diskrepanzen/{json_filename}")
    
    if response:
        st.subheader("Analyse der Lieferzeit:")
        st.info(response)
    else:
        st.info("ℹ️ Keine RAG-Analyse erforderlich.")

    total_cost_after_rag = calculate_rag_total_cost()
    rag_cost = total_cost_after_rag - total_cost_before_rag

    st.header("💰 Kostenübersicht")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("API-Kosten (vor RAG)", f"${total_cost_before_rag:.4f}")
    with col2:
        st.metric("API-Kosten (für RAG)", f"${rag_cost:.4f}")
    with col3:
        st.metric("Gesamtkosten", f"${total_cost_after_rag:.4f}")

    end_time = time.time()
    total_time_taken = end_time - start_time

    st.header("📊 Prozessstatistiken")
    st.metric("Prozessdauer", f"{total_time_taken:.2f} Sekunden")

# Streamlit-Benutzeroberfläche
st.set_page_config(page_title="Rechnungsverarbeitung", page_icon="📊", layout="wide")

st.title("📄 Rechnungsverarbeitung und Vertragsanalyse")
st.write("Wählen Sie eine der folgenden Rechnungen aus, um den Verarbeitungs- und Analyseprozess zu starten:")

# Benutzerdefiniertes CSS für besser aussehende Buttons und Links
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        height: 60px;
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .doc-link {
        display: inline-block;
        padding: 5px 10px;
        margin: 5px;
        background-color: #f0f2f6;
        border-radius: 5px;
        text-decoration: none;
        color: #0066cc;
        font-size: 14px;
        transition: background-color 0.3s;
    }
    .doc-link:hover {
        background-color: #e0e2e6;
    }
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

def create_invoice_section(col, file_name, invoice_link, order_link):
    with col:
        if st.button(f'🧾 {file_name}'):
            main(file_name)
        st.markdown(f'<a href="{invoice_link}" target="_blank" class="doc-link">📄 Rechnung anzeigen</a>', unsafe_allow_html=True)
        st.markdown(f'<a href="{order_link}" target="_blank" class="doc-link">📦 Bestellung anzeigen</a>', unsafe_allow_html=True)

create_invoice_section(
    col1,
    "RE-2024-JUL-27-0001.pdf",
    "https://github.com/LinusLangner/BA_Linus_Langner/blob/main/documents/Lieferantenrechnungen/RE-2024-JUL-27-0001.pdf",
    "https://github.com/LinusLangner/BA_Linus_Langner/blob/main/documents/Bestellaufträge/ON-12345.pdf"
)

create_invoice_section(
    col2,
    "RE-2024-SEP-05-0003.pdf",
    "https://github.com/LinusLangner/BA_Linus_Langner/blob/main/documents/Lieferantenrechnungen/RE-2024-SEP-05-0003.pdf",
    "https://github.com/LinusLangner/BA_Linus_Langner/blob/main/documents/Bestellaufträge/ON-56789.pdf"
)

create_invoice_section(
    col3,
    "INV-2024-11335.pdf",
    "https://github.com/LinusLangner/BA_Linus_Langner/blob/main/documents/Lieferantenrechnungen/INV-2024-11335.pdf",
    "https://github.com/LinusLangner/BA_Linus_Langner/blob/main/documents/Bestellaufträge/PO-2024-006.pdf"
)