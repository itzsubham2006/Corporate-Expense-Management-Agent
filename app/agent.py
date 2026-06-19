from pydantic import BaseModel, Field
from google.adk.workflow import Workflow, node
from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.apps import App
from google.genai import types

class ExpenseReviewDecision(BaseModel):
    category: str = Field(description="The category of the expense (e.g. Travel, Office Supplies, Meals).")
    amount: float = Field(description="The estimated amount of the expense, or 0.0 if not specified.")
    requires_approval: bool = Field(description="True if the expense is high (e.g. > $100) or ambiguous and requires human approval, False otherwise.")
    reason: str = Field(description="Reason for requiring approval or not.")

classifier = LlmAgent(
    name="classifier",
    model="gemini-3.5-flash",
    instruction="Classify the expense request. Determine the category, amount, and whether it requires approval (e.g. if amount > $100 or if the category is high risk).",
    output_schema=ExpenseReviewDecision,
)

@node(rerun_on_resume=True)
async def review_expense(ctx: Context, node_input: dict):
    requires_approval = node_input.get("requires_approval", False)
    category = node_input.get("category", "Unknown")
    amount = node_input.get("amount", 0.0)
    reason = node_input.get("reason", "No reason provided")

    if requires_approval:
        if not ctx.resume_inputs or "review_decision" not in ctx.resume_inputs:
            yield RequestInput(
                interrupt_id="review_decision",
                message=(
                    f"⚠️ Expense approval requested for {category} of ${amount:.2f}.\n"
                    f"Reason: {reason}\n"
                    f"Please reply with 'approve' or 'reject' to proceed."
                )
            )
            return

        decision = ctx.resume_inputs["review_decision"].strip().lower()
        if "approve" in decision:
            yield Event(
                output=f"Approved expense: {category} (${amount:.2f})",
                route="approved",
                state={"approved": True}
            )
        else:
            yield Event(
                output=f"Rejected expense: {category} (${amount:.2f})",
                route="rejected",
                state={"approved": False}
            )
    else:
        yield Event(
            output=f"Auto-approved expense: {category} (${amount:.2f})",
            route="approved",
            state={"approved": True}
        )

def process_approved(node_input: str):
    message = f"✅ Success: {node_input}. The expense has been successfully logged."
    yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=message)]))
    yield Event(output=message)

def process_rejected(node_input: str):
    message = f"❌ Cancelled: {node_input}. The expense was rejected."
    yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=message)]))
    yield Event(output=message)

root_agent = Workflow(
    name="root_agent",
    description="An expense management workflow with automatic classification and human-in-the-loop review.",
    edges=[
        ('START', classifier),
        (classifier, review_expense),
        (review_expense, {
            "approved": process_approved,
            "rejected": process_rejected
        })
    ]
)

app = App(root_agent=root_agent, name="app")
