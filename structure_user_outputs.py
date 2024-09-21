def print_user_formatted_data(order_structured_data, invoice_structured_data):
    print(f"Strukturierte Daten der Bestellung:")
    print(f"Bestellnummer: {order_structured_data.order_number}")
    for idx, article in enumerate(order_structured_data.article_number):
        print(f"Artikel {idx + 1}:")
        print(f"  Artikelnummer : {article}")
        print(f"  Beschreibung  : {order_structured_data.description[idx]}")
        print(f"  Menge         : {order_structured_data.amount[idx]}")
        print(f"  Preis         : {order_structured_data.prices[idx]}")
    print(f"Gesamtbetrag   : {order_structured_data.total}")
    print(f"Datum          : {order_structured_data.date}")
    
    print(f"\nStrukturierte Daten der Rechnung:")
    print(f"Bestellnummer: {invoice_structured_data.order_number}")
    for idx, article in enumerate(invoice_structured_data.article_number):
        print(f"Artikel {idx + 1}:")
        print(f"  Artikelnummer : {article}")
        print(f"  Beschreibung  : {invoice_structured_data.description[idx]}")
        print(f"  Menge         : {invoice_structured_data.amount[idx]}")
        print(f"  Preis         : {invoice_structured_data.prices[idx]}")
    print(f"Gesamtbetrag   : {invoice_structured_data.total}")
    print(f"Datum          : {invoice_structured_data.date}\n\n")


def generate_informative_text(discrepancies):
    output_text = ""

    if discrepancies['article_number_discrepancies']:
        output_text += "\nAbweichungen bei den Artikelnummern:\n"
        for entry in discrepancies['article_number_discrepancies']:
            output_text += f"  Artikelnummer in der Rechnung: {entry['invoice_article_number']} stimmt nicht mit den Artikelnummern in der Bestellung überein: {entry['order_article_numbers']}\n"

    if discrepancies['quantity_discrepancies']:
        output_text += "\nAbweichungen bei der Menge:\n"
        for entry in discrepancies['quantity_discrepancies']:
            output_text += f"  Artikelnummer: {entry['article_number']}\n"
            output_text += f"  Bestellte Menge : {entry['order_amount']}\n"
            output_text += f"  Gelieferte Menge: {entry['invoice_amount']}\n"
            output_text += f"  Unterschied    : {entry['difference']}\n"

    if discrepancies['price_discrepancies']:
        output_text += "\nAbweichungen bei den Preisen:\n"
        for entry in discrepancies['price_discrepancies']:
            output_text += f"  Artikelnummer: {entry['article_number']}\n"
            output_text += f"  Bestellpreis : {entry['order_price']}\n"
            output_text += f"  Rechnungspreis: {entry['invoice_price']}\n"
            output_text += f"  Unterschied   : {entry['difference']}\n"

    if discrepancies['description_discrepancies']:
        output_text += "\nAbweichungen bei den Beschreibungen:\n"
        for entry in discrepancies['description_discrepancies']:
            output_text += f"  Artikelnummer: {entry['article_number']}\n"
            output_text += f"  Beschreibung in der Bestellung: {entry['order_description']}\n"
            output_text += f"  Beschreibung in der Rechnung  : {entry['invoice_description']}\n"
            output_text += f"  Anmerkung   : {entry['note']}\n"

    if discrepancies['date_discrepancies']:
        output_text += "\nAbweichungen bei den Daten:\n"
        for entry in discrepancies['date_discrepancies']:
            output_text += f"  Bestelldatum       : {entry['order_date']}\n"
            output_text += f"  Rechnungsdatum     : {entry['invoice_date']}\n"
            output_text += f"  Unterschied in Tagen: {entry['difference_in_days']}\n"

    return output_text



