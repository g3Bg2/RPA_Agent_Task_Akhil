#!/usr/bin/env python3

import os
import sys
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

try:
    import pdfplumber
    import pandas as pd
    import easyocr  # Fallback OCR if pytesseract unavailable
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install -r requirements.txt")
    sys.exit(1)

try:
    from config.settings import (
        INVOICES_FOLDER,
        PROCESSED_FOLDER,
        EXCEPTIONS_FOLDER,
        REVIEW_FOLDER,
        LOGS_FOLDER,
        CONFIG_FOLDER,
        PO_DATA_FILE,
        REPORT_FILE,
        AMOUNT_TOLERANCE,
        DATE_TOLERANCE,
        HIGH_RISK_VENDORS,
        HIGH_VALUE_THRESHOLD,
        LOG_FILE_NORMAL,
        LOG_FORMAT,
        LOG_LEVEL,
    )
    LOG_FILE = LOG_FILE_NORMAL
except ImportError:
    print("Please ensure config/settings.py exists and .env file is configured")
    sys.exit(1)

def setup_logging():
    """Configure logging to file and console."""
    os.makedirs(LOGS_FOLDER, exist_ok=True)
    
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger

logger = setup_logging()

def initialize_directories():
    """Create required directories if they don't exist."""
    directories = [
        INVOICES_FOLDER,
        PROCESSED_FOLDER,
        EXCEPTIONS_FOLDER,
        REVIEW_FOLDER,
        LOGS_FOLDER,
        CONFIG_FOLDER
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    logger.info("All required directories initialized.")

def intake_stage(watch_mode=False, watch_interval=10):
    """
    Stage 1: Monitor a folder for new invoice PDFs.
    
    Args:
        watch_mode (bool): If True, continuously monitor. If False, process once.
        watch_interval (int): Seconds between checks (if watch_mode=True).
    
    Returns:
        List[Path]: Paths to new PDF files found.
    """
    logger.info("=== STAGE 1: INTAKE ===")
    logger.info(f"Monitoring folder: {INVOICES_FOLDER}")
    
    if not os.path.exists(INVOICES_FOLDER):
        logger.warning(f"Invoices folder does not exist: {INVOICES_FOLDER}")
        return []
    
    # Find all PDFs in the invoices folder
    pdf_files = list(Path(INVOICES_FOLDER).glob("*.pdf"))
    
    if pdf_files:
        logger.info(f"Found {len(pdf_files)} PDF(s) to process.")
    else:
        logger.info("No PDFs found in invoices folder.")
    
    return pdf_files

def extract_text_from_pdf(pdf_path: str, use_easyocr=True) -> str:
    """
    Extract text from PDF using pdfplumber and fallback OCR if needed.
    
    Args:
        pdf_path (str): Path to PDF file.
        use_easyocr (bool): Use EasyOCR if pdfplumber fails.
    
    Returns:
        str: Extracted text from PDF.
    """
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        logger.warning(f"pdfplumber extraction failed for {pdf_path}: {e}")
        
        # Fallback: Use EasyOCR
        if use_easyocr:
            try:
                logger.info(f"Attempting OCR extraction with EasyOCR for {pdf_path}")
                reader = easyocr.Reader(['en'])
                # Convert PDF to images first (requires pdf2image)
                # For now, we'll log a warning
                logger.warning("EasyOCR requires PDF-to-image conversion. Please install pdf2image.")
            except Exception as ocr_error:
                logger.error(f"EasyOCR extraction failed: {ocr_error}")
    
    return text

def extract_invoice_data(pdf_path: str) -> Dict[str, Optional[str]]:
    """
    Extract key invoice fields from PDF.
    
    Args:
        pdf_path (str): Path to PDF file.
    
    Returns:
        Dict: Extracted data with keys: invoice_number, vendor_name, amount, date.
    """
    logger.info(f"Extracting data from: {pdf_path}")
    
    extracted_data = {
        "invoice_number": None,
        "vendor_name": None,
        "amount": None,
        "date": None,
        "pdf_path": pdf_path
    }
    
    try:
        text = extract_text_from_pdf(pdf_path)
        
        if not text.strip():
            logger.warning(f"No text extracted from {pdf_path}")
            return extracted_data
        
        # Pattern matching for key fields
        # Invoice Number: Match patterns like "INV001", "Invoice #001", etc.
        invoice_pattern = r"(?:INV\s+)?(?:INV|Invoice\s*#?\s*)(\d{3,})"
        invoice_match = re.search(invoice_pattern, text, re.IGNORECASE)
        if invoice_match:
            extracted_data["invoice_number"] = f"INV{invoice_match.group(1).zfill(3)}"
        
        # Vendor Name: Look for common patterns (Vendor:, From:, Bill From:)
        vendor_pattern = r"(?:Vendor|From|Bill\s+From)[\s:]+([A-Za-z\s&.,]+?)(?:\n|Invoice|Date)"
        vendor_match = re.search(vendor_pattern, text, re.IGNORECASE)
        if vendor_match:
            extracted_data["vendor_name"] = vendor_match.group(1).strip()
        
        # Amount: Look for currency patterns - Total Amount, Grand Total, Amount
        # Look for "Total Amount:" or similar at the end
        amount_pattern = r"(?:Total\s+Amount|Grand\s+Total|Amount\s+Due)[\s:$]*\$?(\d+\.?\d*)"
        amount_match = re.search(amount_pattern, text, re.IGNORECASE)
        if amount_match:
            try:
                extracted_data["amount"] = float(amount_match.group(1))
            except ValueError:
                logger.warning(f"Could not convert amount to float: {amount_match.group(1)}")
        
        # Date: Look for date patterns - Invoice Date: YYYY-MM-DD, Date: MM/DD/YYYY, etc.
        date_pattern = r"(?:Invoice\s+Date|Date|Dated)[\s:]+(\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"
        date_match = re.search(date_pattern, text, re.IGNORECASE)
        if date_match:
            extracted_data["date"] = date_match.group(1)
        
        logger.info(f"Extracted data: {extracted_data}")
        
    except Exception as e:
        logger.error(f"Error extracting data from {pdf_path}: {e}")
    
    return extracted_data

def load_po_data(po_file: str) -> pd.DataFrame:
    """
    Load PO data from CSV.
    
    Args:
        po_file (str): Path to PO CSV file.
    
    Returns:
        pd.DataFrame: PO data.
    """
    try:
        po_df = pd.read_csv(po_file)
        logger.info(f"Loaded PO data: {len(po_df)} records")
        return po_df
    except Exception as e:
        logger.error(f"Error loading PO data: {e}")
        return pd.DataFrame()

def parse_date(date_str: str) -> Optional[datetime]:
    """Parse date string in various formats."""
    if not date_str:
        return None
    
    formats = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    
    logger.warning(f"Could not parse date: {date_str}")
    return None

def validate_invoice_data(
    extracted_data: Dict,
    po_df: pd.DataFrame
) -> Tuple[bool, str]:
    """
    Validate extracted invoice data against PO data.
    
    Args:
        extracted_data (Dict): Extracted invoice fields.
        po_df (pd.DataFrame): PO data.
    
    Returns:
        Tuple[bool, str]: (is_valid, validation_message)
    """
    logger.info("=== STAGE 3: VALIDATE ===")
    
    invoice_number = extracted_data.get("invoice_number")
    vendor_name = extracted_data.get("vendor_name")
    amount = extracted_data.get("amount")
    date_str = extracted_data.get("date")
    
    # Check if invoice number found
    if not invoice_number:
        msg = "Invoice number not found in PDF"
        logger.warning(msg)
        return False, msg
    
    # Find matching PO record
    po_record = po_df[po_df["PO_Number"] == invoice_number]
    
    if po_record.empty:
        msg = f"No matching PO found for invoice number: {invoice_number}"
        logger.warning(msg)
        return False, msg
    
    po_row = po_record.iloc[0]
    validation_details = []
    all_valid = True
    
    # Validate Vendor Name
    if vendor_name:
        expected_vendor = po_row["Vendor_Name"]
        if vendor_name.lower().strip() != expected_vendor.lower().strip():
            all_valid = False
            validation_details.append(
                f"Vendor mismatch: Extracted '{vendor_name}' vs Expected '{expected_vendor}'"
            )
        else:
            logger.info(f"Vendor name matches: {vendor_name}")
    else:
        all_valid = False
        validation_details.append("Vendor name not extracted")
    
    # Validate Amount (±5% tolerance)
    if amount is not None:
        expected_amount = po_row["Expected_Amount"]
        tolerance_range = expected_amount * AMOUNT_TOLERANCE
        lower_bound = expected_amount - tolerance_range
        upper_bound = expected_amount + tolerance_range
        
        if lower_bound <= amount <= upper_bound:
            logger.info(f"Amount within tolerance: {amount} (expected: {expected_amount})")
        else:
            all_valid = False
            validation_details.append(
                f"Amount mismatch: Extracted ${amount:.2f} vs Expected ${expected_amount:.2f} "
                f"(tolerance: ±${tolerance_range:.2f})"
            )
    else:
        all_valid = False
        validation_details.append("Amount not extracted")
    
    # Validate Date (±3 days tolerance)
    if date_str:
        extracted_date = parse_date(date_str)
        expected_date_str = po_row["Expected_Date"]
        expected_date = parse_date(str(expected_date_str))
        
        if extracted_date and expected_date:
            date_diff = abs((extracted_date - expected_date).days)
            if date_diff <= DATE_TOLERANCE:
                logger.info(f"Date within tolerance: {extracted_date.date()} (expected: {expected_date.date()})")
            else:
                all_valid = False
                validation_details.append(
                    f"Date mismatch: Extracted {extracted_date.date()} vs Expected {expected_date.date()} "
                    f"(difference: {date_diff} days, tolerance: ±{DATE_TOLERANCE} days)"
                )
        else:
            all_valid = False
            validation_details.append(f"Could not parse dates for comparison")
    else:
        all_valid = False
        validation_details.append("Date not extracted")
    
    validation_message = " | ".join(validation_details) if validation_details else "All validations passed"
    
    logger.info(f"Validation result: {'PASS' if all_valid else 'FAIL'} - {validation_message}")
    return all_valid, validation_message

def send_notification(notification_type: str, subject: str, body: str):
    """
    Send a simulated notification (console log or email stub).
    
    Args:
        notification_type (str): "email", "slack", or "console"
        subject (str): Notification subject.
        body (str): Notification body.
    """
    if notification_type == "console" or notification_type == "email":
        logger.info(f"[NOTIFICATION] {subject}\n{body}")
    elif notification_type == "slack":
        logger.info(f"[SLACK MESSAGE] {subject}\n{body}")

def decision_agent(
    extracted_data: Dict,
    is_valid: bool,
    validation_message: str,
    po_df: pd.DataFrame
) -> Tuple[str, str]:
    """
    Intelligent decision agent to route invoices.
    
    Routes invoices to:
    - "processed": Valid and low risk
    - "exceptions": Validation failed
    - "review": High value (>$1000) or high-risk vendor
    
    Args:
        extracted_data (Dict): Extracted invoice data.
        is_valid (bool): Validation result.
        validation_message (str): Validation details.
        po_df (pd.DataFrame): PO data (for lookup).
    
    Returns:
        Tuple[str, str]: (destination_folder, status_description)
    """
    logger.info("=== STAGE 4: DECIDE (Decision Agent) ===")
    
    destination = "exceptions"
    status = "Exception"
    
    # Check high-value threshold
    amount = extracted_data.get("amount")
    vendor = extracted_data.get("vendor_name", "").strip()
    
    if is_valid:
        # Additional decision rules
        if vendor and any(risk_vendor.lower() in vendor.lower() for risk_vendor in HIGH_RISK_VENDORS):
            destination = "review"
            status = "Review (High-risk vendor)"
            logger.info(f"Routing to review: High-risk vendor detected - {vendor}")
        elif amount and amount > HIGH_VALUE_THRESHOLD:
            destination = "review"
            status = "Review (High-value invoice)"
            logger.info(f"Routing to review: High-value invoice (${amount:.2f})")
        else:
            destination = "processed"
            status = "Matched"
            logger.info("Invoice validation passed. Routing to processed folder.")
    else:
        destination = "exceptions"
        status = f"Exception ({validation_message})"
        logger.info(f"Validation failed. Routing to exceptions folder.")
        
        # Send notification for exceptions
        send_notification(
            "console",
            "Invoice Validation Failed",
            f"Invoice: {extracted_data.get('invoice_number', 'UNKNOWN')}\n"
            f"Vendor: {extracted_data.get('vendor_name', 'UNKNOWN')}\n"
            f"Reason: {validation_message}"
        )
    
    return destination, status

def move_file_to_folder(source_path: str, destination_folder: str) -> bool:
    """
    Move a file to the destination folder.
    
    Args:
        source_path (str): Full path to file.
        destination_folder (str): Destination folder name (processed/exceptions/review).
    
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        folder_map = {
            "processed": PROCESSED_FOLDER,
            "exceptions": EXCEPTIONS_FOLDER,
            "review": REVIEW_FOLDER
        }
        
        dest_path = folder_map.get(destination_folder)
        if not dest_path:
            logger.error(f"Invalid destination folder: {destination_folder}")
            return False
        
        os.makedirs(dest_path, exist_ok=True)
        filename = os.path.basename(source_path)
        dest_file = dest_path / filename
        
        shutil.move(source_path, str(dest_file))
        logger.info(f"Moved {filename} to {destination_folder} folder")
        return True
    except Exception as e:
        logger.error(f"Error moving file {source_path}: {e}")
        return False

def generate_report(processing_results: List[Dict]):
    """
    Generate a summary report CSV.
    
    Args:
        processing_results (List[Dict]): List of processed invoice records.
    """
    logger.info("=== STAGE 5: REPORT ===")
    
    if not processing_results:
        logger.warning("No processing results to report.")
        return
    
    try:
        report_df = pd.DataFrame(processing_results)
        report_df.to_csv(REPORT_FILE, index=False)
        logger.info(f"Report generated: {REPORT_FILE}")
        logger.info(f"\nReport Summary:\n{report_df.to_string()}\n")
    except Exception as e:
        logger.error(f"Error generating report: {e}")

def process_invoice(pdf_path: str, po_df: pd.DataFrame) -> Dict:
    """
    Process a single invoice through all stages.
    
    Args:
        pdf_path (str): Path to invoice PDF.
        po_df (pd.DataFrame): PO data.
    
    Returns:
        Dict: Processing result record.
    """
    logger.info(f"\n{'='*60}\nProcessing: {os.path.basename(pdf_path)}\n{'='*60}")
    
    # Stage 2: Extract
    extracted_data = extract_invoice_data(str(pdf_path))
    
    # Stage 3: Validate
    is_valid, validation_message = validate_invoice_data(extracted_data, po_df)
    
    # Stage 4: Decide
    destination, status = decision_agent(extracted_data, is_valid, validation_message, po_df)
    move_file_to_folder(str(pdf_path), destination)
    
    # Record result
    result = {
        "Invoice_File_Name": os.path.basename(pdf_path),
        "Invoice_Number": extracted_data.get("invoice_number") or "UNKNOWN",
        "Vendor_Name": extracted_data.get("vendor_name") or "UNKNOWN",
        "Amount": extracted_data.get("amount") or "NOT_EXTRACTED",
        "Status": status,
        "Validation_Details": validation_message,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return result

def main():
    logger.info("\n" + "="*70)
    logger.info("INVOICE AUTOMATION WORKFLOW - START")
    logger.info("="*70)
    
    initialize_directories()
    
    # Load PO data
    po_df = load_po_data(str(PO_DATA_FILE))
    if po_df.empty:
        logger.error("PO data not loaded. Exiting.")
        return
    
    processing_results = []
    
    try:
        # Stage 1: Intake
        pdf_files = intake_stage(watch_mode=False)
        
        if not pdf_files:
            logger.info("No PDFs to process.")
            logger.info("="*70)
            return
        
        # Process each invoice
        for pdf_path in pdf_files:
            result = process_invoice(str(pdf_path), po_df)
            processing_results.append(result)
        
        # Stage 5: Report
        generate_report(processing_results)
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("PROCESSING SUMMARY")
        logger.info("="*70)
        matched_count = sum(1 for r in processing_results if "Matched" in r["Status"])
        exception_count = sum(1 for r in processing_results if "Exception" in r["Status"])
        review_count = sum(1 for r in processing_results if "Review" in r["Status"])
        
        logger.info(f"Total Processed: {len(processing_results)}")
        logger.info(f"  ✓ Matched: {matched_count}")
        logger.info(f"  ! Exceptions: {exception_count}")
        logger.info(f"  ⚠ Review: {review_count}")
        logger.info("="*70 + "\n")
        
    except Exception as e:
        logger.error(f"Unexpected error in main workflow: {e}", exc_info=True)
    finally:
        logger.info("INVOICE AUTOMATION WORKFLOW - END\n")

if __name__ == "__main__":
    main()