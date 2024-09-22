import os
import fitz # PyMuPDF
from pydantic import BaseModel
from openai import OpenAI
from token_tracker import track_token_usage

client = OpenAI()

model = "gpt-4o-2024-08-06"

class DocumentExtraction(BaseModel):
    order_number: str
    article_number: list[str]
    description: list[str]
    amount: list[int]
    prices: list[float]
    total: float
    date: int

def process_order_and_invoice(invoice_pdf_filename):
    # Hilfsfunktion zum Erstellen des PDF-Pfads für die Bestellung
    def create_pdf_path_for_order(order_number):
        pdf_path = f"documents/Bestellaufträge/{order_number}.pdf"
        if os.path.exists(pdf_path):
            print(f"\nDer Bestellauftrag {order_number} wurde im System gefunden.")
        else:
            print(f"\nZu der angegebenen Bestellnummer: {order_number} konnte im System kein Bestellauftrag gefunden werden.\n")
        return pdf_path

    # Hilfsfunktion zum Extrahieren von Text aus einem PDF-Dokument
    def extract_text_from_pdf(pdf_path):
        print(f"\nExtrahiere Text aus PDF: {pdf_path}\n")
        pdf_document = fitz.open(pdf_path)
        text = ""
        for page in pdf_document:
            text += page.get_text("text")
        pdf_document.close()
        print("Text erfolgreich extrahiert.")
        return text

    # Hilfsfunktion zum Extrahieren von strukturierten Daten aus einem unstrukturierten Text
    def extract_structured_data(user_input):
        
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": """Du bist ein Experte für die Extraktion von strukturierten Daten. 
                                                 Du erhältst unstrukturierten Text aus einer Bestellung und sollst 
                                                 diesen in eine vorgegebene Struktur umwandeln."""},
                {"role": "user", "content":     user_input}
            ],
            response_format=DocumentExtraction,
            temperature=0.0
        )

        # Verfolgung der Token-Nutzung
        track_token_usage(completion.usage, model)

        structured_data = completion.choices[0].message.parsed
        return structured_data

    # Schritt 1: Extrahieren und Verarbeiten der Rechnung
    invoice_path = f"documents/Lieferantenrechnungen/{invoice_pdf_filename}"
    invoice_content = extract_text_from_pdf(invoice_path)
    invoice_structured_data = extract_structured_data(invoice_content)
    print("\nExtrahierte strukturierte Daten der Rechnung:\n")
    print(invoice_structured_data)

    # Schritt 2: Extrahieren und Verarbeiten der zugehörigen Bestellung
    order_number = invoice_structured_data.order_number
    order_path = create_pdf_path_for_order(order_number)
    order_content = extract_text_from_pdf(order_path)
    order_structured_data = extract_structured_data(order_content)
    print("\nExtrahierte strukturierte Daten der Bestellung:\n")
    print(order_structured_data)

    return order_structured_data, invoice_structured_data
