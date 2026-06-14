import re 
from rapidfuzz import fuzz

def extract_data(results: list[list[dict[str, int, int]]]):
    #KEYWORDS = {
    #    "final_total": ["grand total", "total amount", "balance due", "total sales inclusive"],
    #    "subtotal": ["subtotal", "total sales exluding", "before tax", "net amount"],
    #    "taxes": ["vat", "tax", "total gst"],
    #    "date": ["date"],
    #    "document_no": ["invoice no", "doc no", "receipt no", "document number", "reference no"],
    #    "customer_id": ["customer id", "member id", "gst id", "account no"],
    #}
    KEYWORDS = {
        "final_total": ["grand total", "total"],
        "subtotal": ["subtotal", "total sales exluding"],
        "taxes": ["vat", "tax"],
        "date": ["date"],
        "document_no": ["invoice no", "doc no", "receipt no", "document number", "reference no"],
        "customer_id": ["customer id", "member id", "gst id", "account no"],
    }
    final_list = []

    for token in results:
        best_category = None
        best_score = 0

        for category, keywords in KEYWORDS.items():
            for kw in keywords:
                score = fuzz.partial_ratio(token.lower(), kw)

                if score > best_score:
                    best_score = score
                    best_category = category

        if best_score > 85:
            final_list.append({
                "category": best_category,
                "score": best_score,
                "position": i,
                "token": token
            })

    print(final_list)