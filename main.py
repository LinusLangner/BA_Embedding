import os
import json
import time  # Importiere das Modul "time", um die Ausführungszeit zu verfolgen
from extract_structured_data import process_order_and_invoice
from compare_data import comparer
from token_tracker import calculate_total_cost
from rag_integration import process_rag_query_from_json, calculate_rag_total_cost  # Importiere RAG-Prozesse

def main(invoice_pdf_filename):
    start_time = time.time()  # Erfasse die Startzeit
    
    # Verarbeite die Bestellung und die Rechnung, um strukturierte Daten zu extrahieren
    print("\nDie Bestellung und die Rechnung werden verarbeitet, um strukturierte Daten zu extrahieren...\n")
    order_structured_data, invoice_structured_data = process_order_and_invoice(invoice_pdf_filename)
    print("\nDie Extraktion der strukturierten Daten ist abgeschlossen.\n")

    # Vergleiche die Bestellung und die Rechnung, um Abweichungen zu identifizieren
    discrepancies = comparer(order_structured_data, invoice_structured_data)
    print("\nGefundene Abweichungen:")
    print(discrepancies)

    # Generiere einen dynamischen Dateinamen für die JSON-Ausgabe
    invoice_base_name = os.path.splitext(invoice_pdf_filename)[0]  # Entfernt die .pdf-Erweiterung
    order_number = invoice_structured_data.order_number
    json_filename = f"{invoice_base_name}_{order_number}.json"

    # Speichere die Abweichungen als JSON-Datei mit dem dynamischen Namen
    with open(f"documents/Diskrepanzen/{json_filename}", "w") as f:
        json.dump(discrepancies, f, indent=4)
    
    print(f"\nDie Abweichungen wurden in der Datei {json_filename} gespeichert.")


    # Berechne und gib die Gesamtkosten für die Token-Nutzung (vor RAG) aus
    total_cost_before_rag = calculate_total_cost()
    print(f"\nGesamtkosten der API-Nutzung (vor RAG): ${total_cost_before_rag}")

    # Starte den RAG-Prozess am Ende mit der gespeicherten JSON-Datei der Abweichungen
    print("\nStarte den RAG-Prozess zur Überprüfung der Lieferzeit...\n")
    response = process_rag_query_from_json(f"documents/Diskrepanzen/{json_filename}")
    print(f"\nAntwort des LLMs:\n{response}")

    # Berechne und gib die Gesamtkosten für die Token-Nutzung nach dem RAG-Prozess aus
    total_cost_after_rag = calculate_rag_total_cost()
    rag_cost = total_cost_after_rag - total_cost_before_rag
    print(f"\nGesamtkosten der API-Nutzung (nur für RAG): ${rag_cost}")
    print(f"\nGesamtkosten der API-Nutzung (inklusive RAG): ${total_cost_after_rag}")

    # Erfasse die Endzeit und berechne die Gesamtdauer des Prozesses
    end_time = time.time()
    total_time_taken = end_time - start_time
    print(f"\nGesamtdauer des Prozesses: {total_time_taken:.2f} Sekunden")

if __name__ == "__main__":
    main("RE-2024-SEP-05-0003.pdf")
