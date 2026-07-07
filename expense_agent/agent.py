import base64
import json
import re
from typing import Any
from pydantic import BaseModel, Field

from google.adk.workflow import Workflow, node
from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.apps import App
from google.genai import types

from expense_agent.config import THRESHOLD_USD, MODEL_NAME

# Regexes for scrubbing PII
SSN_REGEX = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
CC_REGEX = re.compile(r'\b(?:\d[ -]*?){13,16}\b')

# Prompt Injection Keywords
INJECTION_KEYWORDS = [
    "ignore previous instructions",
    "ignore instructions",
    "ignore rules",
    "bypass rules",
    "system prompt",
    "auto-approve",
    "autoapprove",
    "override rules",
    "bypass threshold",
    "force approval",
    "you must approve",
    "approved: true"
]


class ExpenseRiskAssessment(BaseModel):
    risk_level: str = Field(description="The risk level of the expense: LOW, MEDIUM, or HIGH.")
    risk_factors: list[str] = Field(description="List of risk factors identified (e.g. suspicious category, excessive description, unusual request).")
    justification: str = Field(description="Detailed explanation/justification for the risk rating.")


def parse_expense_input(node_input: Any) -> dict:
    """Robust parser for extracting expense details from base64/JSON strings or dicts."""
    raw_str = ""
    if isinstance(node_input, dict):
        raw_str = json.dumps(node_input)
    elif isinstance(node_input, str):
        raw_str = node_input
    elif hasattr(node_input, "parts") and node_input.parts:
        raw_str = node_input.parts[0].text
    elif hasattr(node_input, "text"):
        raw_str = node_input.text
    else:
        raw_str = str(node_input)

    try:
        data_dict = json.loads(raw_str)
    except Exception:
        # Not JSON, treat as raw text description
        return {
            "amount": 0.0,
            "submitter": "Unknown",
            "category": "Unknown",
            "description": raw_str,
            "date": "Unknown"
        }

    # If wrapped in a Pub/Sub "message" structure
    if "message" in data_dict and isinstance(data_dict["message"], dict):
        data_dict = data_dict["message"]

    data_val = data_dict.get("data")
    if data_val is None:
        expense_data = data_dict
    else:
        if isinstance(data_val, str):
            try:
                decoded = base64.b64decode(data_val).decode('utf-8')
                expense_data = json.loads(decoded)
            except Exception:
                try:
                    expense_data = json.loads(data_val)
                except Exception:
                    expense_data = {"description": data_val}
        elif isinstance(data_val, dict):
            expense_data = data_val
        else:
            expense_data = {}

    amount_raw = expense_data.get("amount", 0.0)
    try:
        amount = float(amount_raw)
    except Exception:
        amount = 0.0

    return {
        "amount": amount,
        "submitter": expense_data.get("submitter", expense_data.get("user", "Unknown")),
        "category": expense_data.get("category", "Unknown"),
        "description": expense_data.get("description", ""),
        "date": expense_data.get("date", "Unknown")
    }


@node
def parse_input(node_input: Any) -> Event:
    expense = parse_expense_input(node_input)
    amount = expense["amount"]
    
    if amount < THRESHOLD_USD:
        return Event(
            output=expense,
            route="auto_approve",
            state={"expense": expense}
        )
    else:
        return Event(
            output=expense,
            route="security_check",
            state={"expense": expense}
        )


@node
def auto_approve(node_input: dict):
    submitter = node_input.get("submitter", "Unknown")
    amount = node_input.get("amount", 0.0)
    category = node_input.get("category", "Unknown")
    
    message = f"✅ Auto-Approved: Expense of ${amount:.2f} for '{category}' submitted by {submitter} is under the ${THRESHOLD_USD:.2f} threshold. No human or LLM review required."
    
    yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=message)]))
    yield Event(
        output={
            "status": "APPROVED",
            "message": message,
            "amount": amount,
            "submitter": submitter
        }
    )


@node
def security_check(node_input: dict) -> Event:
    description = node_input.get("description", "")
    redacted_categories = []
    
    # 1. PII Scrubbing
    scrubbed_desc = description
    if SSN_REGEX.search(scrubbed_desc):
        scrubbed_desc = SSN_REGEX.sub("[REDACTED-SSN]", scrubbed_desc)
        redacted_categories.append("SSN")
        
    if CC_REGEX.search(scrubbed_desc):
        scrubbed_desc = CC_REGEX.sub("[REDACTED-CC]", scrubbed_desc)
        redacted_categories.append("Credit Card")
        
    clean_expense = dict(node_input)
    clean_expense["description"] = scrubbed_desc
    clean_expense["redacted_categories"] = redacted_categories
    
    # 2. Prompt Injection Check
    desc_lower = scrubbed_desc.lower()
    has_injection = any(kw in desc_lower for kw in INJECTION_KEYWORDS)
    
    if has_injection:
        clean_expense["prompt_injection_detected"] = True
        return Event(
            output=clean_expense,
            route="security_flagged",
            state={"expense": clean_expense}
        )
    else:
        clean_expense["prompt_injection_detected"] = False
        return Event(
            output=clean_expense,
            route="clean_expense",
            state={"expense": clean_expense}
        )


llm_risk_review = LlmAgent(
    name="llm_risk_review",
    model=MODEL_NAME,
    instruction=(
        "You are an expert financial risk auditor. Review the expense report details (amount, category, description) "
        "and determine if there are any risk factors or suspicious details. "
        "Provide a risk level (LOW, MEDIUM, HIGH) and list specific risk factors."
    ),
    output_schema=ExpenseRiskAssessment,
    output_key="risk_assessment",
)


@node(rerun_on_resume=True)
async def human_review(ctx: Context, node_input: Any):
    expense = ctx.state.get("expense", {})
    risk_assessment = ctx.state.get("risk_assessment")
    
    amount = expense.get("amount", 0.0)
    submitter = expense.get("submitter", "Unknown")
    category = expense.get("category", "Unknown")
    description = expense.get("description", "")
    redacted = expense.get("redacted_categories", [])
    prompt_injection = expense.get("prompt_injection_detected", False)
    
    details = (
        f"📋 Expense Details:\n"
        f"  - Submitter: {submitter}\n"
        f"  - Amount: ${amount:.2f}\n"
        f"  - Category: {category}\n"
        f"  - Description: {description}\n"
    )
    if redacted:
        details += f"  - 🔒 PII Redacted: {', '.join(redacted)}\n"
        
    if prompt_injection:
        details += (
            f"\n🚨 SECURITY ALERT: Prompt injection attempt detected in description!\n"
            f"  - LLM risk review was bypassed.\n"
            f"  - Flagged as a Security Event.\n"
        )
    elif risk_assessment:
        risk_level = risk_assessment.get("risk_level", "UNKNOWN")
        risk_factors = risk_assessment.get("risk_factors", [])
        justification = risk_assessment.get("justification", "No justification provided.")
        
        details += (
            f"\n🤖 LLM Risk Review:\n"
            f"  - Risk Level: {risk_level}\n"
            f"  - Risk Factors: {', '.join(risk_factors) if risk_factors else 'None'}\n"
            f"  - Justification: {justification}\n"
        )
        
    if not ctx.resume_inputs or "decision" not in ctx.resume_inputs:
        yield RequestInput(
            interrupt_id="decision",
            message=(
                f"{details}\n"
                f"Please reply with 'approve' or 'reject' to make a decision."
            )
        )
        return

    decision_val = ctx.resume_inputs["decision"]
    if isinstance(decision_val, dict):
        user_decision = (decision_val.get("decision") or "").strip().lower()
    else:
        user_decision = decision_val.strip().lower()
    
    result_payload = {
        "expense": expense,
        "risk_assessment": risk_assessment,
        "security_event": prompt_injection,
        "redacted_pii": redacted
    }
    
    if "approve" in user_decision:
        yield Event(
            output=result_payload,
            route="approved"
        )
    else:
        yield Event(
            output=result_payload,
            route="rejected"
        )


def process_approved(node_input: dict):
    expense = node_input["expense"]
    amount = expense.get("amount", 0.0)
    submitter = expense.get("submitter", "Unknown")
    category = expense.get("category", "Unknown")
    
    msg = f"✅ Expense Approved: ${amount:.2f} for '{category}' submitted by {submitter} has been approved by the reviewer."
    yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=msg)]))
    yield Event(output={"status": "APPROVED", "message": msg, "details": node_input})


def process_rejected(node_input: dict):
    expense = node_input["expense"]
    amount = expense.get("amount", 0.0)
    submitter = expense.get("submitter", "Unknown")
    category = expense.get("category", "Unknown")
    
    msg = f"❌ Expense Rejected: ${amount:.2f} for '{category}' submitted by {submitter} has been rejected by the reviewer."
    yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=msg)]))
    yield Event(output={"status": "REJECTED", "message": msg, "details": node_input})


root_agent = Workflow(
    name="root_agent",
    description="An ambient expense approval agent with PII scrubbing, injection defense, LLM risk review, and HITL verification.",
    edges=[
        ('START', parse_input),
        (parse_input, {
            "auto_approve": auto_approve,
            "security_check": security_check
        }),
        (security_check, {
            "security_flagged": human_review,
            "clean_expense": llm_risk_review
        }),
        (llm_risk_review, human_review),
        (human_review, {
            "approved": process_approved,
            "rejected": process_rejected
        })
    ]
)

app = App(root_agent=root_agent, name="expense_agent")
