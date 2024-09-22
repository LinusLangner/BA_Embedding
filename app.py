import streamlit as st
import os
import json
import time
from extract_structured_data import process_order_and_invoice
from compare_data import comparer
from structure_user_outputs import print_user_formatted_data
from token_tracker import calculate_total_cost
from rag_integration import process_rag_query_from_json, calculate_rag_total_cost

# Define the main process function
def main(invoice_pdf_filename):
    start_time = time.time()

    st.write("\nDie Bestellung und die Rechnung werden verarbeitet, um strukturierte Daten zu extrahieren...\n")
    order_structured_data, invoice_structured_data = process_order_and_invoice(invoice_pdf_filename)
    st.write("\nDie Extraktion der strukturierten Daten ist abgeschlossen.\n")

    st.write("Strukturierte Daten:")
    print_user_formatted_data(order_structured_data, invoice_structured_data)

    discrepancies = comparer(order_structured_data, invoice_structured_data)
    st.write("\nGefundene Abweichungen:")
    st.write(discrepancies)

    invoice_base_name = os.path.splitext(invoice_pdf_filename)[0]
    order_number = invoice_structured_data.order_number
    json_filename = f"{invoice_base_name}_{order_number}.json"

    with open(f"documents/Diskrepanzen/{json_filename}", "w") as f:
        json.dump(discrepancies, f, indent=4)

    st.write(f"\nDie Abweichungen wurden in der Datei {json_filename} gespeichert.")

    total_cost_before_rag = calculate_total_cost()
    st.write(f"\nGesamtkosten der API-Nutzung (vor RAG): ${total_cost_before_rag}")

    st.write("\nStarte den RAG-Prozess zur Überprüfung der Lieferzeit...\n")
    response = process_rag_query_from_json(f"documents/Diskrepanzen/{json_filename}")
    st.write(f"\nAntwort des LLMs:\n{response}")

    total_cost_after_rag = calculate_rag_total_cost()
    rag_cost = total_cost_after_rag - total_cost_before_rag
    st.write(f"\nGesamtkosten der API-Nutzung (nur für RAG): ${rag_cost}")
    st.write(f"\nGesamtkosten der API-Nutzung (inklusive RAG): ${total_cost_after_rag}")

    end_time = time.time()
    total_time_taken = end_time - start_time
    st.write(f"\nGesamtdauer des Prozesses: {total_time_taken:.2f} Sekunden")

# Streamlit UI
st.title("Rechnungsverarbeitung und RAG-Prozess")

# Buttons for the three PDF files
if st.button('RE-2024-JUL-27-0001.pdf'):
    main("RE-2024-JUL-27-0001.pdf")

if st.button('RE-2024-SEP-05-0003.pdf'):
    main("RE-2024-SEP-05-0003.pdf")

if st.button('INV-2024-11335.pdf'):
    main("INV-2024-11335.pdf")
