# Invoice Data Extraction & Validation Automation

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager
- (For Agentic) Ollama local LLM service running on localhost:11434

python3 -m venv .venv

source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install Python packages
pip install -r requirements.txt

## Quick Start

### Generate Sample Invoices

```bash
python create_sample_invoices.py
```

This creates 3 test PDFs in the `invoices/` folder:
- **INV001.pdf** - Valid (TechSupplies Inc, $1500.00) - Routes to Review
- **INV002.pdf** - Valid (Office Essentials Ltd, $2500.00) - Routes to Review
- **INV003.pdf** - Amount mismatch ($3500 vs expected $3200) - Routes to Exceptions
- **INV004.pdf** - Valid low-value invoice ($200.00) - Processed directly

### Run the Automation Workflow

```bash
# Standard mode: Process all PDFs in invoices/ folder once
python invoice_automation.py
```

---

## 🧠 Running the Agentic Workflow

The **Agentic Workflow** uses LangChain agents with local Ollama models for intelligent decision-making.

### Prerequisites for Agentic Mode
1. **Ollama service** running: `ollama serve`
2. **Model installed**: mistral:latest or qwen3:0.6b
3. **LangChain packages** installed (in requirements.txt)

### Quick Start

```bash
# 1. Start Ollama in a separate terminal
ollama serve

# 2. Run agentic workflow
python invoice_automation_agentic.py

# 3. Check results
cat report.csv
ls review/       # High-value invoices
ls exceptions/   # Failed validations
```