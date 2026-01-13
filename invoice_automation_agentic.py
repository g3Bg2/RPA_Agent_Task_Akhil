#!/usr/bin/env python3

import os
import sys
import json
import logging
import shutil
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
from langchain_core.tools import Tool
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
import pdfplumber

try:
    from config.settings import (
        INVOICES_FOLDER,
        PROCESSED_FOLDER,
        EXCEPTIONS_FOLDER,
        REVIEW_FOLDER,
        LOGS_FOLDER,
        PO_DATA_FILE,
        REPORT_FILE,
        AMOUNT_TOLERANCE,
        DATE_TOLERANCE,
        HIGH_VALUE_THRESHOLD,
        VENDOR_RISK_THRESHOLD,
        OLLAMA_MODEL,
        OLLAMA_BASE_URL,
        LOG_FILE_AGENTIC,
        LOG_LEVEL,
    )
    LOG_FILE = LOG_FILE_AGENTIC
except ImportError:
    print("Please ensure config/settings.py exists and .env file is configured")
    sys.exit(1)

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class InvoiceData:
    pdf_path: str
    invoice_number: Optional[str] = None
    vendor_name: Optional[str] = None
    amount: Optional[float] = None
    date: Optional[str] = None
    extraction_confidence: float = 0.0

@dataclass
class ValidationResult:
    is_valid: bool
    vendor_match: bool = False
    amount_match: bool = False
    date_match: bool = False
    vendor_score: float = 0.0
    amount_variance: float = 0.0
    date_variance: int = 0
    details: str = ""

class InvoiceStatus(Enum):
    MATCHED = "Matched"
    EXCEPTION = "Exception"
    REVIEW = "Review"
    HIGH_RISK = "High-Risk"
    PENDING = "Pending"

@dataclass
class ProcessingResult:
    pdf_filename: str
    invoice_number: str
    vendor_name: str
    amount: float
    status: InvoiceStatus
    decision_reason: str
    destination_folder: str
    timestamp: str

# Agent Tools

class InvoiceTools:
    
    def __init__(self):
        self.po_cache = {}
        self.vendor_risk_cache = {}
        self._load_po_cache()
    
    def _load_po_cache(self):

        try:
            po_df = pd.read_csv(PO_DATA_FILE)
            for idx, row in po_df.iterrows():
                self.po_cache[row['PO_Number']] = {
                    'vendor': row['Vendor_Name'],
                    'amount': row['Expected_Amount'],
                    'date': row['Expected_Date']
                }
            logger.info(f"Loaded {len(self.po_cache)} PO records into cache")
        except Exception as e:
            logger.error(f"Error loading PO cache: {e}")
    
    def extract_invoice_fields(self, pdf_path: str) -> str:
        """
        Extract invoice fields from PDF.
        
        Tool Input: pdf_path (string)
        Tool Output: JSON string with extracted fields
        """
        try:
            data = InvoiceData(pdf_path=pdf_path)
            
            with pdfplumber.open(pdf_path) as pdf:
                text = "\n".join([page.extract_text() or "" for page in pdf.pages])
            
            # Invoice Number extraction
            inv_pattern = r"(?:INV\s+)?(?:INV|Invoice\s*#?\s*)(\d{3,})"
            inv_match = re.search(inv_pattern, text, re.IGNORECASE)
            if inv_match:
                data.invoice_number = f"INV{inv_match.group(1).zfill(3)}"
            
            # Vendor extraction
            vendor_pattern = r"(?:Vendor|From|Bill\s+From)[\s:]+([A-Za-z\s&.,]+?)(?:\n|Invoice|Date)"
            vendor_match = re.search(vendor_pattern, text, re.IGNORECASE)
            if vendor_match:
                data.vendor_name = vendor_match.group(1).strip()
            
            # Amount extraction
            amount_pattern = r"(?:Total\s+Amount|Grand\s+Total|Amount\s+Due)[\s:$]*\$?(\d+\.?\d*)"
            amount_match = re.search(amount_pattern, text, re.IGNORECASE)
            if amount_match:
                data.amount = float(amount_match.group(1))
            
            # Date extraction
            date_pattern = r"(?:Invoice\s+Date|Date|Dated)[\s:]+(\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"
            date_match = re.search(date_pattern, text, re.IGNORECASE)
            if date_match:
                data.date = date_match.group(1)
            
            data.extraction_confidence = 0.9 if (data.invoice_number and data.vendor_name and data.amount) else 0.5
            
            logger.info(f"Extracted from {Path(pdf_path).name}: {data.invoice_number}")
            return json.dumps(asdict(data), default=str)
        
        except Exception as e:
            logger.error(f"Extraction error for {pdf_path}: {e}")
            return json.dumps({"error": str(e), "pdf_path": pdf_path})
    
    def retrieve_po_data(self, invoice_number: str) -> str:
        """
        Retrieve PO data for given invoice number.
        
        Tool Input: invoice_number (string)
        Tool Output: JSON string with PO data or 'not_found'
        """
        try:
            if invoice_number in self.po_cache:
                po_data = self.po_cache[invoice_number]
                logger.info(f"Retrieved PO for {invoice_number}")
                return json.dumps({"found": True, "po": po_data, "invoice_number": invoice_number})
            else:
                logger.warning(f"PO not found for {invoice_number}")
                return json.dumps({"found": False, "invoice_number": invoice_number, "message": "PO not found in database"})
        
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return json.dumps({"error": str(e)})
    
    def calculate_vendor_risk_score(self, vendor_name: str) -> str:
        """
        Calculate vendor risk score using LLM reasoning.
        
        Tool Input: vendor_name (string)
        Tool Output: JSON with risk_score (0-1) and reasoning
        """
        try:
            # Check cache first
            if vendor_name in self.vendor_risk_cache:
                return json.dumps(self.vendor_risk_cache[vendor_name])
            
            # High-risk vendor list
            high_risk = ["BlackList Vendor", "Suspicious Corp", "Unknown Vendor"]
            
            if any(risk_vendor.lower() in vendor_name.lower() for risk_vendor in high_risk):
                score = 0.9
                reason = f"Vendor '{vendor_name}' matches high-risk list"
            elif vendor_name and len(vendor_name) > 3:
                score = 0.2
                reason = f"Vendor '{vendor_name}' is established and verified"
            else:
                score = 0.7
                reason = f"Vendor '{vendor_name}' requires additional verification"
            
            result = {
                "vendor_name": vendor_name,
                "risk_score": score,
                "is_high_risk": score >= VENDOR_RISK_THRESHOLD,
                "reasoning": reason
            }
            
            self.vendor_risk_cache[vendor_name] = result
            logger.info(f"Risk score for {vendor_name}: {score}")
            return json.dumps(result)
        
        except Exception as e:
            logger.error(f"Risk calculation error: {e}")
            return json.dumps({"error": str(e)})
    
    def validate_against_po(self, invoice_data_json: str, po_data_json: str) -> str:
        """
        Validate extracted invoice against PO data.
        
        Tool Input: JSON strings for invoice_data and po_data
        Tool Output: JSON with validation result
        """
        try:
            invoice = json.loads(invoice_data_json)
            po = json.loads(po_data_json)
            
            result = ValidationResult(is_valid=True)
            
            # Vendor validation
            if invoice.get('vendor_name'):
                expected_vendor = po['po'].get('vendor', '')
                if invoice['vendor_name'].lower().strip() == expected_vendor.lower().strip():
                    result.vendor_match = True
                    result.vendor_score = 1.0
                else:
                    result.is_valid = False
                    result.vendor_match = False
                    result.vendor_score = 0.3
            
            # Amount validation
            if invoice.get('amount') is not None:
                expected_amount = po['po'].get('amount', 0)
                tolerance_range = expected_amount * AMOUNT_TOLERANCE
                if abs(invoice['amount'] - expected_amount) <= tolerance_range:
                    result.amount_match = True
                    result.amount_variance = 0.0
                else:
                    result.is_valid = False
                    result.amount_match = False
                    result.amount_variance = ((invoice['amount'] - expected_amount) / expected_amount * 100)
            
            # Date validation
            if invoice.get('date'):
                expected_date = po['po'].get('date', '')
                try:
                    from datetime import datetime
                    inv_date = datetime.strptime(invoice['date'], '%Y-%m-%d')
                    exp_date = datetime.strptime(expected_date, '%Y-%m-%d')
                    date_diff = abs((inv_date - exp_date).days)
                    if date_diff <= DATE_TOLERANCE:
                        result.date_match = True
                        result.date_variance = 0
                    else:
                        result.is_valid = False
                        result.date_match = False
                        result.date_variance = date_diff
                except Exception as e:
                    logger.warning(f"Date parsing error: {e}")
            
            logger.info(f"Validation: {'PASS' if result.is_valid else 'FAIL'}")
            return json.dumps(asdict(result), default=str)
        
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return json.dumps({"error": str(e)})
    
    def move_file(self, source_path: str, destination: str) -> str:
        """
        Move processed file to destination folder.
        
        Tool Input: source_path, destination ('processed', 'exceptions', 'review')
        Tool Output: JSON with success status
        """
        try:
            folder_map = {
                "processed": PROCESSED_FOLDER,
                "exceptions": EXCEPTIONS_FOLDER,
                "review": REVIEW_FOLDER
            }
            
            dest_folder = folder_map.get(destination)
            if not dest_folder:
                return json.dumps({"success": False, "error": f"Invalid destination: {destination}"})
            
            os.makedirs(dest_folder, exist_ok=True)
            filename = os.path.basename(source_path)
            dest_file = dest_folder / filename
            
            shutil.move(source_path, str(dest_file))
            logger.info(f"Moved to {destination}/")
            return json.dumps({"success": True, "destination": str(dest_file)})
        
        except Exception as e:
            logger.error(f"File move error: {e}")
            return json.dumps({"success": False, "error": str(e)})
    
    def get_available_tools(self) -> List[Tool]:
        """Return all tools for agent"""
        return [
            Tool(
                name="extract_invoice_fields",
                func=self.extract_invoice_fields,
                description="Extract invoice number, vendor, amount, and date from PDF file"
            ),
            Tool(
                name="retrieve_po_data",
                func=self.retrieve_po_data,
                description="Retrieve purchase order data for an invoice number from database"
            ),
            Tool(
                name="calculate_vendor_risk_score",
                func=self.calculate_vendor_risk_score,
                description="Calculate vendor risk score (0-1) based on vendor name and history"
            ),
            Tool(
                name="validate_against_po",
                func=self.validate_against_po,
                description="Validate extracted invoice data against PO data with tolerance checking"
            ),
            Tool(
                name="move_file",
                func=self.move_file,
                description="Move processed invoice file to destination folder (processed/exceptions/review)"
            ),
        ]


# Agent Nodes

class InvoiceAgentNode:
    """Base class for agent processing nodes"""
    
    def __init__(self, name: str, description: str, model: OllamaLLM, tools: List[Tool]):
        self.name = name
        self.description = description
        self.llm = model
        self.tools = tools
        self.agent = None
        self.executor = None
        self._setup_agent()
    
    def _setup_agent(self):
        """Setup ReAct agent with tools"""
        try:
            prompt = PromptTemplate.from_template(
                """You are an invoice processing agent. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
{agent_scratchpad}"""
            )
            
            self.agent = create_react_agent(self.llm, self.tools, prompt)
            self.executor = AgentExecutor.from_agent_and_tools(
                agent=self.agent,
                tools=self.tools,
                verbose=False,
                max_iterations=3,
                handle_parsing_errors=True,
                early_stopping_method="force"
            )
            logger.info(f"Agent '{self.name}' initialized successfully")
        except Exception as e:
            logger.error(f"Error setting up agent: {e}")
            self.executor = None
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input through agent"""
        raise NotImplementedError("Subclasses must implement process()")

class IntakeNode(InvoiceAgentNode):
    """Stage 1: Intake - Discover PDF files"""
    
    def __init__(self, model: OllamaLLM, tools: List[Tool]):
        super().__init__(
            name="Intake",
            description="Monitor folder for new invoice PDFs",
            model=model,
            tools=tools
        )
    
    def process(self, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Find and return invoice PDFs"""
        pdf_files = list(Path(INVOICES_FOLDER).glob("*.pdf"))
        logger.info(f"=== STAGE 1: INTAKE ===")
        logger.info(f"Found {len(pdf_files)} PDF(s)")
        return {
            "stage": "intake",
            "pdf_files": [str(f) for f in pdf_files],
            "count": len(pdf_files)
        }

class ExtractNode(InvoiceAgentNode):
    """Stage 2: Extract - OCR field extraction using agent"""
    
    def __init__(self, model: OllamaLLM, tools: List[Tool]):
        super().__init__(
            name="Extract",
            description="Extract invoice fields using OCR and LLM",
            model=model,
            tools=tools
        )
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract invoice data from PDF"""
        pdf_path = input_data.get("pdf_path")
        logger.info(f"=== STAGE 2: EXTRACT ===")
        logger.info(f"Processing: {Path(pdf_path).name}")
        
        result_json = self.tools[0].func(pdf_path)
        extracted = json.loads(result_json)
        
        logger.info(f"Extracted: {extracted.get('invoice_number')}")
        return {"stage": "extract", "data": extracted}

class ValidationNode(InvoiceAgentNode):
    """Stage 3: Validate - PO comparison using agent reasoning"""
    
    def __init__(self, model: OllamaLLM, tools: List[Tool]):
        super().__init__(
            name="Validate",
            description="Validate invoice against PO with intelligent comparison",
            model=model,
            tools=tools
        )
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate extracted data"""
        invoice_number = input_data.get("invoice_number")
        logger.info(f"=== STAGE 3: VALIDATE ===")
        
        po_result_json = self.tools[1].func(invoice_number)
        po_result = json.loads(po_result_json)
        
        if po_result.get("found"):
            # Validate against PO
            validation_json = self.tools[3].func(json.dumps(input_data), po_result_json)
            result = json.loads(validation_json)
        else:
            result = {"is_valid": False, "error": "PO not found"}
        
        logger.info(f"Validation: {'PASS' if result.get('is_valid') else 'FAIL'}")
        return {"stage": "validate", "result": result}

class DecisionNode(InvoiceAgentNode):
    """Stage 4: Decision Agent - Intelligent routing based on reasoning"""
    
    def __init__(self, model: OllamaLLM, tools: List[Tool]):
        super().__init__(
            name="Decision",
            description="Intelligently route invoices using agentic reasoning",
            model=model,
            tools=tools
        )
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make intelligent routing decision"""
        logger.info(f"=== STAGE 4: DECISION AGENT ===")
        
        vendor_name = input_data.get('vendor_name', '')
        risk_json = self.tools[2].func(vendor_name)
        risk_data = json.loads(risk_json)
        
        # Decision logic using agent reasoning
        decision_prompt = f"""
Based on the following invoice and validation data, decide routing:

Invoice: {input_data.get('invoice_number')}
Vendor: {vendor_name}
Amount: ${input_data.get('amount', 0)} (Threshold: ${HIGH_VALUE_THRESHOLD})
Valid: {input_data.get('is_valid', False)}
Vendor Risk: {risk_data.get('risk_score', 0.5)}

Rules:
- "processed": Valid + Low-Risk (<0.7) + Low-Value (<{HIGH_VALUE_THRESHOLD})
- "review": Valid but (High-Risk OR High-Value)
- "exceptions": Invalid"""
        
        if self.executor:
            try:
                decision_result = self.executor.invoke({
                    "input": decision_prompt
                })
                decision_text = decision_result.get("output", "")
            except:
                decision_text = ""
        else:
            decision_text = ""
        
        # Parse decision
        is_valid = input_data.get('is_valid', False)
        is_high_risk = risk_data.get('is_high_risk', False)
        is_high_value = input_data.get('amount', 0) >= HIGH_VALUE_THRESHOLD
        
        if is_valid and not is_high_risk and not is_high_value:
            destination = "processed"
            status = "Matched"
        elif is_valid and (is_high_risk or is_high_value):
            destination = "review"
            status = "Review"
        else:
            destination = "exceptions"
            status = "Exception"
        
        logger.info(f"Decision: {destination} - {status}")
        
        return {
            "stage": "decide",
            "destination": destination,
            "status": status,
            "reasoning": f"Valid:{is_valid} HighRisk:{is_high_risk} HighValue:{is_high_value}"
        }

class ReportNode(InvoiceAgentNode):
    """Stage 5: Report - Generate summary"""
    
    def __init__(self, model: OllamaLLM, tools: List[Tool]):
        super().__init__(
            name="Report",
            description="Generate processing summary report",
            model=model,
            tools=tools
        )
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate report"""
        logger.info(f"=== STAGE 5: REPORT ===")
        
        results = input_data.get("results", [])
        report_path = REPORT_FILE
        
        try:
            df = pd.DataFrame(results)
            df.to_csv(report_path, index=False)
            logger.info(f"Report generated: {report_path}")
            logger.info(f"\nReport Summary:\n{df.to_string()}\n")
            return {"stage": "report", "success": True, "file": str(report_path)}
        except Exception as e:
            logger.error(f"Report generation error: {e}")
            return {"stage": "report", "success": False, "error": str(e)}

class InvoiceAgentOrchestrator:
    """Orchestrates multiple agents in workflow pipeline"""
    
    def __init__(self, model_name: str = OLLAMA_MODEL):
        self.model_name = model_name
        self.llm = OllamaLLM(model=model_name, base_url=OLLAMA_BASE_URL)
        self.tools = InvoiceTools()
        self.tool_list = self.tools.get_available_tools()
        self.setup_nodes()
        
        logger.info(f"Initialized orchestrator with model: {model_name}")
    
    def setup_nodes(self):
        """Initialize all agentic nodes"""
        self.intake_node = IntakeNode(self.llm, self.tool_list)
        self.extract_node = ExtractNode(self.llm, self.tool_list)
        self.validate_node = ValidationNode(self.llm, self.tool_list)
        self.decision_node = DecisionNode(self.llm, self.tool_list)
        self.report_node = ReportNode(self.llm, self.tool_list)
    
    def process_invoice(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        """Process single invoice through all stages"""
        logger.info(f"\n{'='*60}\nProcessing: {Path(pdf_path).name}\n{'='*60}")
        
        try:
            
            # Stage 2: Extract
            extract_result = self.extract_node.process({"pdf_path": pdf_path})
            if "error" in extract_result.get("data", {}):
                logger.error(f"Extraction failed: {extract_result['data']['error']}")
                return None
            
            invoice_data = extract_result["data"]
            
            # Stage 3: Validate
            validate_result = self.validate_node.process(invoice_data)
            is_valid = validate_result.get("result", {}).get("is_valid", False)
            
            # Stage 4: Decide (Decision Agent)
            decision_data = {**invoice_data, "is_valid": is_valid}
            decision_result = self.decision_node.process(decision_data)
            
            destination = decision_result.get("destination")
            status = decision_result.get("status")
            
            # Move file
            self.tools.move_file(pdf_path, destination)
            
            # Create result record
            result = {
                "Invoice_File_Name": Path(pdf_path).name,
                "Invoice_Number": invoice_data.get("invoice_number", "UNKNOWN"),
                "Vendor_Name": invoice_data.get("vendor_name", "UNKNOWN"),
                "Amount": invoice_data.get("amount", 0),
                "Status": status,
                "Validation_Details": validate_result.get("result", {}).get("details", "N/A"),
                "Decision_Reasoning": decision_result.get("reasoning", ""),
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            return None
    
    def run_workflow(self) -> List[Dict[str, Any]]:
        """Execute complete agentic workflow"""
        logger.info("\n" + "="*70)
        logger.info("INVOICE AUTOMATION WORKFLOW - AGENTIC (LangChain + Ollama)")
        logger.info("="*70)
        
        # Initialize directories
        os.makedirs(INVOICES_FOLDER, exist_ok=True)
        os.makedirs(PROCESSED_FOLDER, exist_ok=True)
        os.makedirs(EXCEPTIONS_FOLDER, exist_ok=True)
        os.makedirs(REVIEW_FOLDER, exist_ok=True)
        os.makedirs(LOGS_FOLDER, exist_ok=True)
        
        # Stage 1: Intake
        intake_result = self.intake_node.process()
        pdf_files = intake_result.get("pdf_files", [])
        
        if not pdf_files:
            logger.info("No PDFs to process.")
            return []
        
        # Process each invoice
        processing_results = []
        for pdf_path in pdf_files:
            result = self.process_invoice(pdf_path)
            if result:
                processing_results.append(result)
        
        # Stage 5: Report
        if processing_results:
            self.report_node.process({"results": processing_results})
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("PROCESSING SUMMARY")
        logger.info("="*70)
        matched = sum(1 for r in processing_results if "Matched" in r.get("Status", ""))
        review = sum(1 for r in processing_results if "Review" in r.get("Status", ""))
        exception = sum(1 for r in processing_results if "Exception" in r.get("Status", ""))
        
        logger.info(f"Total Processed: {len(processing_results)}")
        logger.info(f"  ✓ Matched: {matched}")
        logger.info(f"  ⚠ Review: {review}")
        logger.info(f"  ✗ Exceptions: {exception}")
        logger.info("="*70 + "\n")
        
        return processing_results

def main():
    """Main entry point"""
    try:
        orchestrator = InvoiceAgentOrchestrator(model_name=OLLAMA_MODEL)
        results = orchestrator.run_workflow()
        logger.info("Workflow completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
