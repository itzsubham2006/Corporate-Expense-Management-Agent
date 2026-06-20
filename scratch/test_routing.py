import asyncio
from typing import Any
from google.adk.workflow import Workflow, node
from google.adk.agents.context import Context
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

@node
async def parse_expense(ctx: Context, node_input: Any):
    print("--- parse_expense node ---")
    if isinstance(node_input, types.Content):
        text = "".join(part.text for part in node_input.parts if part.text is not None)
    else:
        text = str(node_input)
    amount = float(text)
    if amount < 100:
        yield Event(output={"amount": amount, "msg": "under 100"}, route="auto_approve")
    else:
        yield Event(output={"amount": amount, "msg": "100 or more"}, route="llm_review")

@node
async def auto_approve(ctx: Context, node_input: dict):
    print("--- auto_approve node ---")
    print("Received:", node_input)
    yield Event(output=f"Auto-approved: {node_input['amount']}")

@node
async def risk_review(ctx: Context, node_input: dict):
    print("--- risk_review node ---")
    print("Received:", node_input)
    yield Event(output=f"Risk review needed: {node_input['amount']}")

workflow = Workflow(
    name="expense_test",
    edges=[
        ('START', parse_expense),
        (parse_expense, {
            "auto_approve": auto_approve,
            "llm_review": risk_review
        })
    ]
)

async def main():
    session_service = InMemorySessionService()
    session = await session_service.create_session(user_id="test_user", app_name="test")
    runner = Runner(agent=workflow, session_service=session_service, app_name="test")
    
    print("=== Run 1: Amount 50 ===")
    msg = types.Content(role="user", parts=[types.Part.from_text(text="50")])
    async for event in runner.run_async(new_message=msg, user_id="test_user", session_id=session.id):
        print("Event:", event.output)

    print("=== Run 2: Amount 150 ===")
    msg = types.Content(role="user", parts=[types.Part.from_text(text="150")])
    async for event in runner.run_async(new_message=msg, user_id="test_user", session_id=session.id):
        print("Event:", event.output)

asyncio.run(main())
