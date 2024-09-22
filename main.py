# main.py
import os
import json
import time  # Import the time module to track execution time
from extract_structured_data import process_order_and_invoice
from compare_data import comparer
from structure_user_outputs import generate_informative_text, print_user_formatted_data
from token_tracker import calculate_total_cost  # Import the token cost calculator
from rag_integration import process_rag_query_from_json, calculate_rag_total_cost  # Import RAG processes

def process_invoice(invoice_pdf_filename):
    start_time = time.time()  # Capture the start time

    # Process the order and invoice to extract structured data
    order_structured_data, invoice_structured_data = process_order_and_invoice(invoice_pdf_filename)
    
    # Compare the order and invoice to identify discrepancies
    discrepancies = comparer(order_structured_data, invoice_structured_data)

    # Generate a dynamic filename for the JSON output
    invoice_base_name = os.path.splitext(invoice_pdf_filename)[0]  # Removes the .pdf extension
    order_number = invoice_structured_data.order_number
    json_filename = f"{invoice_base_name}_{order_number}.json"

    # Save the discrepancies as a JSON file with the dynamic name
    with open(f"documents/Diskrepanzen/{json_filename}", "w") as f:
        json.dump(discrepancies, f, indent=4)
    
    # Generate and print an informative text based on prioritized discrepancies
    informative_text = generate_informative_text(discrepancies)

    # Calculate the total token usage cost (before RAG)
    total_cost_before_rag = calculate_total_cost()

    # Start the RAG process at the end with the saved JSON discrepancies file
    response = process_rag_query_from_json(f"documents/Diskrepanzen/{json_filename}")

    # Calculate the total token usage cost after the RAG process
    total_cost_after_rag = calculate_rag_total_cost()
    rag_cost = total_cost_after_rag - total_cost_before_rag  # Cost only for the RAG process

    # Capture the end time and calculate the total time taken
    end_time = time.time()
    total_time_taken = end_time - start_time

    # Collect all the results in a dictionary and return
    return {
        "informative_text": informative_text,
        "total_cost_before_rag": total_cost_before_rag,
        "rag_response": response,
        "rag_cost": rag_cost,
        "total_cost_after_rag": total_cost_after_rag,
        "total_time_taken": total_time_taken,
    }
