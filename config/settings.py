import os
import logging
from pathlib import Path
from dotenv import load_dotenv

ENV_FILE = Path(__file__).parent.parent / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
else:
    EXAMPLE_ENV = Path(__file__).parent.parent / ".env.example"
    if EXAMPLE_ENV.exists():
        print(f"No .env file found. Copy .env.example to .env and update values.")

BASE_DIR = Path(os.getenv("BASE_DIR", ".")).resolve()
if not BASE_DIR.is_absolute():
    BASE_DIR = Path(__file__).parent.parent / BASE_DIR

if not BASE_DIR.is_absolute():
    BASE_DIR = BASE_DIR.resolve()

def _get_path(env_var: str, default_name: str) -> Path:
    """Helper to construct absolute paths"""
    value = os.getenv(env_var, default_name)
    path = Path(value)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path

INVOICES_FOLDER = _get_path("INVOICES_FOLDER", "invoices")
PROCESSED_FOLDER = _get_path("PROCESSED_FOLDER", "processed")
EXCEPTIONS_FOLDER = _get_path("EXCEPTIONS_FOLDER", "exceptions")
REVIEW_FOLDER = _get_path("REVIEW_FOLDER", "review")
LOGS_FOLDER = _get_path("LOGS_FOLDER", "logs")
CONFIG_FOLDER = _get_path("CONFIG_FOLDER", "config")


PO_DATA_FILE = _get_path("PO_DATA_FILE", "po_data.csv")
REPORT_FILE = _get_path("REPORT_FILE", "report.csv")

AMOUNT_TOLERANCE = float(os.getenv("AMOUNT_TOLERANCE", "0.05"))
DATE_TOLERANCE = int(os.getenv("DATE_TOLERANCE", "3"))
HIGH_VALUE_THRESHOLD = float(os.getenv("HIGH_VALUE_THRESHOLD", "1000.0"))
VENDOR_RISK_THRESHOLD = float(os.getenv("VENDOR_RISK_THRESHOLD", "0.7"))

HIGH_RISK_VENDORS_STR = os.getenv("HIGH_RISK_VENDORS", "BlackList Vendor,Suspicious Corp,Unknown Vendor")
HIGH_RISK_VENDORS = [v.strip() for v in HIGH_RISK_VENDORS_STR.split(",")]

LOG_LEVEL_STR = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_STR, logging.INFO)
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(levelname)s - %(message)s")

LOG_FILE_NORMAL = _get_path("LOG_FILE_NORMAL", "logs/automation_log.txt")
LOG_FILE_AGENTIC = _get_path("LOG_FILE_AGENTIC", "logs/agentic_automation_log.txt")

LOG_FILE = LOG_FILE_NORMAL

TESSERACT_PATH = os.getenv("TESSERACT_PATH", None)
if TESSERACT_PATH and TESSERACT_PATH.strip():
    TESSERACT_PATH = TESSERACT_PATH.strip()
else:
    TESSERACT_PATH = None

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:latest")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

def print_config():
    """Print current configuration (for debugging)"""
    print("\n" + "="*70)
    print("INVOICE AUTOMATION CONFIGURATION")
    print("="*70)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Invoices Folder: {INVOICES_FOLDER}")
    print(f"Processed Folder: {PROCESSED_FOLDER}")
    print(f"Exceptions Folder: {EXCEPTIONS_FOLDER}")
    print(f"Review Folder: {REVIEW_FOLDER}")
    print(f"Logs Folder: {LOGS_FOLDER}")
    print(f"PO Data File: {PO_DATA_FILE}")
    print(f"Report File: {REPORT_FILE}")
    print(f"Amount Tolerance: {AMOUNT_TOLERANCE*100}%")
    print(f"Date Tolerance: {DATE_TOLERANCE} days")
    print(f"High Value Threshold: ${HIGH_VALUE_THRESHOLD}")
    print(f"High Risk Vendors: {HIGH_RISK_VENDORS}")
    print(f"Log Level: {LOG_LEVEL_STR}")
    print(f"Ollama Model: {OLLAMA_MODEL}")
    print("="*70 + "\n")

if __name__ == "__main__":
    print_config()
