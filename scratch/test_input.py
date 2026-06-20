import asyncio
from typing import Any
from google.adk.workflow import Workflow, node
from google.adk.agents.context import Context
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

@node
async def first_node(ctx: Context, node_input: Any):
    print("--- first_node run ---")
    print("type(node_input):", type(node_input))
    print("node_input:", node_input)
    print("ctx attributes:", dir(ctx))
    print("ctx.state:", ctx.state)
    
    # Extract message text
    text = ""
    if isinstance(node_input, types.Content):
        text = "".join(part.text for part in node_input.parts if part.text is not None)
    elif isinstance(node_input, str):
        text = node_input
    elif isinstance(node_input, dict):
        # Maybe it's a dict
        text = str(node_input)
    print("Extracted text:", text)
    
    yield Event(output=f"Done: {text}")

workflow = Workflow(
    name="test_workflow",
    edges=[
        ('START', first_node)
    ]
)

async def main():
    session_service = InMemorySessionService()
    session = await session_service.create_session(user_id="test_user", app_name="test")
    runner = Runner(agent=workflow, session_service=session_service, app_name="test")
    
    # 1. Test running with string input
    print("--- Running with string input ---")
    msg = types.Content(role="user", parts=[types.Part.from_text(text="hello")])
    async for event in runner.run_async(new_message=msg, user_id="test_user", session_id=session.id):
        print("Event output:", event.output)

    # 2. Test running with JSON event input
    print("--- Running with JSON event input ---")
    json_str = '{"data": "eyBhbW91bnQiOiAxMDB9"}'
    msg = types.Content(role="user", parts=[types.Part.from_text(text=json_str)])
    async for event in runner.run_async(new_message=msg, user_id="test_user", session_id=session.id):
        print("Event output:", event.output)

asyncio.run(main())
