#!/usr/bin/env python3

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from pathlib import Path

def create_sample_invoices():
    
    invoices_folder = Path(__file__).parent / "invoices"
    invoices_folder.mkdir(exist_ok=True)
    
    # Invoice 1: Valid invoice (matches PO)
    invoice_1 = {
        "number": "INV001",
        "vendor": "TechSupplies Inc",
        "amount": 1500.00,  # Expected: 1500.00
        "date": "2026-01-10",
        "items": [
            ("Computer Equipment", 2, 750.00),
            ("Networking Cables", 5, 0.00),
        ]
    }
    
    # Invoice 2: Valid invoice (matches PO)
    invoice_2 = {
        "number": "INV002",
        "vendor": "Office Essentials Ltd",
        "amount": 2500.00,  # Expected: 2500.00
        "date": "2026-01-11",
        "items": [
            ("Office Chairs", 5, 500.00),
            ("Desks", 2, 0.00),
        ]
    }
    
    # Invoice 3: Invalid invoice (amount mismatch)
    invoice_3 = {
        "number": "INV003",
        "vendor": "Cloud Services Corp",
        "amount": 3500.00,
        "date": "2026-01-12",
        "items": [
            ("Cloud Storage (100GB)", 12, 250.00),
            ("Support Services", 1, 0.00),
        ]
    }

    # Invoice 4: Valid invoice (low-value)
    invoice_4 = {
        "number": "INV004",
        "vendor": "Stationery World",
        "amount": 200.00,
        "date": "2026-01-13",
        "items": [
            ("Notebooks", 10, 10.00),
            ("Pens", 20, 5.00),
        ]
    }
    
    invoices = [invoice_1, invoice_2, invoice_3, invoice_4]
    
    for inv in invoices:
        create_invoice_pdf(invoices_folder, inv)

def create_invoice_pdf(folder: Path, invoice_data: dict):
    """Create a single invoice PDF."""
    filename = folder / f"{invoice_data['number']}.pdf"
    
    c = canvas.Canvas(str(filename), pagesize=letter)
    width, height = letter
    
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 50, "INVOICE")
    
    c.setFont("Helvetica", 10)
    y = height - 100
    
    c.drawString(50, y, f"INV {invoice_data['number']}")
    y -= 20
    c.drawString(50, y, f"Vendor: {invoice_data['vendor']}")
    y -= 20
    c.drawString(50, y, f"Invoice Date: {invoice_data['date']}")
    y -= 40
    
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, y, "Description")
    c.drawString(300, y, "Qty")
    c.drawString(350, y, "Unit Price")
    c.drawString(450, y, "Total")
    y -= 20
    
    c.setFont("Helvetica", 9)
    for item, qty, unit_price in invoice_data["items"]:
        c.drawString(50, y, item)
        c.drawString(300, y, str(qty))
        c.drawString(350, y, f"${unit_price:.2f}")
        c.drawString(450, y, f"${qty * unit_price:.2f}")
        y -= 15

    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(350, y, "Total Amount:")
    c.drawString(450, y, f"${invoice_data['amount']:.2f}")
    
    c.save()
    print(f"Created: {filename}")

if __name__ == "__main__":
    create_sample_invoices()
    print("\nSample invoices created in ./invoices/ folder")
