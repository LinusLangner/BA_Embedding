import os
import json
import time  # Import the time module to track execution time
from extract_structured_data import process_order_and_invoice
from compare_data import comparer
from structure_user_outputs import generate_informative_text, print_user_formatted_data
from token_tracker import calculate_total_cost  # Import the token cost calculator
from rag_integration import process_rag_query_from_json, calculate_rag_total_cost  # Import RAG processes

def main(invoice_pdf_filename):
    start_time = time.time()  # Capture the start time
    
    # Process the order and invoice to extract structured data
    print("\nDie Bestellung und die Rechnung werden verarbeitet, um strukturierte Daten zu extrahieren...\n")
    order_structured_data, invoice_structured_data = process_order_and_invoice(invoice_pdf_filename)
    print("\nDie Extraktion der strukturierten Daten ist abgeschlossen.\n")

    # Print the structured data for the order and invoice
    print_user_formatted_data(order_structured_data, invoice_structured_data)

    # Compare the order and invoice to identify discrepancies
    discrepancies = comparer(order_structured_data, invoice_structured_data)
    print("\nGefundene Abweichungen:")
    print(discrepancies)

    # Generate a dynamic filename for the JSON output
    invoice_base_name = os.path.splitext(invoice_pdf_filename)[0]  # Removes the .pdf extension
    order_number = invoice_structured_data.order_number
    json_filename = f"{invoice_base_name}_{order_number}.json"

    # Save the discrepancies as a JSON file with the dynamic name
    with open(f"documents/Diskrepanzen/{json_filename}", "w") as f:
        json.dump(discrepancies, f, indent=4)
    
    print(f"\nDie Abweichungen wurden in der Datei {json_filename} gespeichert.")

    # Generate and print an informative text based on prioritized discrepancies
    informative_text = generate_informative_text(discrepancies)
    print("\nBenutzerfreundlicher informativer Text:")
    print(informative_text)

    # Calculate and print the total token usage cost (before RAG)
    total_cost_before_rag = calculate_total_cost()
    print(f"\nGesamtkosten der API-Nutzung (vor RAG): ${total_cost_before_rag}")

    # Start the RAG process at the end with the saved JSON discrepancies file
    print("\nStarte den RAG-Prozess zur Überprüfung der Lieferzeit...\n")
    response = process_rag_query_from_json(f"documents/Diskrepanzen/{json_filename}")
    print(f"\nAntwort des LLMs:\n{response}")

    # Calculate and print the total token usage cost after the RAG process
    total_cost_after_rag = calculate_rag_total_cost()
    rag_cost = total_cost_after_rag - total_cost_before_rag  # Cost only for the RAG process
    print(f"\nGesamtkosten der API-Nutzung (nur für RAG): ${rag_cost}")
    print(f"\nGesamtkosten der API-Nutzung (inklusive RAG): ${total_cost_after_rag}")

    # Capture the end time and calculate the total time taken
    end_time = time.time()
    total_time_taken = end_time - start_time
    print(f"\nGesamtdauer des Prozesses: {total_time_taken:.2f} Sekunden")

if __name__ == "__main__":
    main("RE-2024-JUL-27-0001.pdf")
