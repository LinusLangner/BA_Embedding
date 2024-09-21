from openai import OpenAI
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime
from token_tracker import track_token_usage

client = OpenAI()

class DocumentExtraction(BaseModel):
    order_number: str
    article_number: List[str]
    description: List[str]
    amount: List[int]
    prices: List[float]
    total: float
    date: int

model = "gpt-4o-mini"

# Funktion zum Vergleich von Artikelnummern mit GPT-4
def compare_article_numbers_with_gpt(order_article_numbers: List[str], invoice_article_numbers: List[str]) -> List[Dict[str, Any]]:
    discrepancies = []
    print("\nVergleiche Artikelnummern...")
    for invoice_article in invoice_article_numbers:
        print(f"Überprüfen, ob die Artikelnummer in der Rechnung: {invoice_article} mit einer der Artikelnummern in der Bestellung übereinstimmt: {order_article_numbers}")
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": """Du bist ein Experte für den Vergleich von Artikelnummern in Bestell- und Rechnungsdaten.
                                                Antworte immer mit 'ja' oder 'nein' (Groß-/Kleinschreibung beachten)"""},

                {"role": "user", "content": f"""Artikelnummer in der Rechnung: {invoice_article}. 
                                                Stimmt diese mit einer der folgenden Artikelnummern in der Bestellung überein: {order_article_numbers}?"""}
            ],
            temperature=0.0
        )

        # Verfolgung der Token-Nutzung
        track_token_usage(response.usage, model)

        gpt_response = response.choices[0].message.content

        print(f"Antwort von GPT-4o: {gpt_response}")

        if gpt_response.lower().startswith("nein"):
            discrepancies.append({
                "invoice_article_number": invoice_article,
                "order_article_numbers": order_article_numbers,
                "note": """Artikelnummer in der Rechnung stimmt nicht 
                mit einer Artikelnummer in der Bestellung überein"""
            })
            
    return discrepancies

# Funktion zum Vergleich der Mengen
def compare_amounts(order_amounts: List[int], invoice_amounts: List[int]) -> List[Dict[str, Any]]:
    discrepancies = []
    print("\nVergleiche Mengen...")
    for i, order_amount in enumerate(order_amounts):
        invoice_amount = invoice_amounts[i] if i < len(invoice_amounts) else None
        print(f"Vergleiche Bestellmenge: {order_amount} mit Rechnungsmenge: {invoice_amount}")
        if order_amount != invoice_amount:
            discrepancies.append({
                "article_number": i + 1,
                "order_amount": order_amount,
                "invoice_amount": invoice_amount,
                "difference": order_amount - invoice_amount if invoice_amount is not None else "Nicht angegeben"
            })
    return discrepancies

# Funktion zum Vergleich der Preise
def compare_prices(order_prices: List[float], invoice_prices: List[float]) -> List[Dict[str, Any]]:
    discrepancies = []
    print("\nVergleiche Preise...")
    for i, order_price in enumerate(order_prices):
        invoice_price = invoice_prices[i] if i < len(invoice_prices) else None
        print(f"Vergleiche Bestellpreis: {order_price} mit Rechnungspreis: {invoice_price}")
        if order_price != invoice_price:
            discrepancies.append({
                "article_number": i + 1,
                "order_price": order_price,
                "invoice_price": invoice_price,
                "difference": order_price - invoice_price if invoice_price is not None else "Nicht angegeben"
            })
    return discrepancies

# Funktion zum Vergleich der Gesamtsummen
def compare_total(order_total: float, invoice_total: float) -> List[Dict[str, Any]]:
    print("\nVergleiche Gesamtsummen...")
    print(f"Bestellsumme: {order_total} vs. Rechnungssumme: {invoice_total}")
    
    discrepancies = []

    if order_total != invoice_total:
        discrepancies.append({
            "order_total": order_total,
            "invoice_total": invoice_total,
            "difference": order_total - invoice_total
        })

    return discrepancies

# Funktion zum Vergleich der Beschreibungen mit GPT-4
def compare_descriptions_with_gpt(order_descriptions: List[str], invoice_descriptions: List[str]) -> List[Dict[str, Any]]:
    discrepancies = []
    print("\nVergleiche Beschreibungen...")
    for i, order_desc in enumerate(order_descriptions):
        invoice_desc = invoice_descriptions[i] if i < len(invoice_descriptions) else None
        print(f"Vergleiche Beschreibung in der Bestellung: {order_desc} mit Beschreibung in der Rechnung: {invoice_desc}")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": """Du bist ein Experte für den Vergleich von Produktbeschreibungen. 
                                                Antworte immer mit 'ja' oder 'nein' (Groß-/Kleinschreibung beachten)."""},

                {"role": "user", "content": f"Beschreiben die folgenden Beschreibungen das selbe Produkt? "
                                            f"Beschreibung in der Bestellung: {order_desc} "
                                            f"Beschreibung in der Rechnung: {invoice_desc}"}
            ],
            temperature=0.0
        )

        # Verfolgung der Token-Nutzung
        track_token_usage(response.usage, model)
        
        gpt_response = response.choices[0].message.content.lower()
        print(f"Antwort von GPT-4: {gpt_response}")
        if gpt_response.lower().startswith("nein"):
            discrepancies.append({
                "article_number": i + 1,
                "order_description": order_desc,
                "invoice_description": invoice_desc,
                "note": "Die Beschreibungen sind unterschiedlich"
            })

    return discrepancies

# Funktion zum Vergleich der Daten
def compare_dates(order_date: int, invoice_date: int) -> List[Dict[str, Any]]:
    print("\nVergleiche Daten...")
    print(f"Vergleiche Bestelldatum: {order_date} mit Rechnungsdatum: {invoice_date}")

    order_date_obj = datetime.strptime(str(order_date), "%Y%m%d")
    invoice_date_obj = datetime.strptime(str(invoice_date), "%Y%m%d")
    difference_in_days = (invoice_date_obj - order_date_obj).days

    discrepancies = [{
        "order_date": order_date,
        "invoice_date": invoice_date,
        "difference_in_days": difference_in_days
    }]
    print(f"Unterschied in Tagen: {difference_in_days}")
    return discrepancies

# Hauptfunktion zum Vergleich der Bestell- und Rechnungsdaten
def comparer(order_structured_data: DocumentExtraction, invoice_structured_data: DocumentExtraction) -> Dict[str, Any]:
    print("Starte den Vergleich der Bestell- und Rechnungsdaten mit GPT-4 und Python...")
    discrepancies = {
        "article_number_discrepancies": compare_article_numbers_with_gpt(order_structured_data.article_number, invoice_structured_data.article_number),
        "quantity_discrepancies": compare_amounts(order_structured_data.amount, invoice_structured_data.amount),
        "price_discrepancies": compare_prices(order_structured_data.prices, invoice_structured_data.prices),
        "total_discrepancies": compare_total(float(order_structured_data.total), float(invoice_structured_data.total)),
        "description_discrepancies": compare_descriptions_with_gpt(order_structured_data.description, invoice_structured_data.description),
        "date_discrepancies": compare_dates(order_structured_data.date, invoice_structured_data.date)
    }
    print("Vergleich abgeschlossen.")
    return discrepancies