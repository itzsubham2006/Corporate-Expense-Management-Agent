import asyncio
from typing import Any
from google.adk.workflow import Workflow, node
from google.adk.agents.context import Context
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

@node
async def test_state_node(ctx: Context, node_input: Any):
    print("ctx.state class:", ctx.state.__class__)
    print("ctx.state dir:", dir(ctx.state))
    # Try setting state directly
    try:
        ctx.state.set("my_key", "my_val")
        print("ctx.state.set worked")
    except Exception as e:
        print("ctx.state.set failed:", e)
        
    try:
        ctx.state["my_key2"] = "my_val2"
        print("ctx.state[key] = val worked")
    except Exception as e:
        print("ctx.state[key] = val failed:", e)
        
    yield Event(output="Done", state={"event_state": "val"})

workflow = Workflow(
    name="test_state_workflow",
    edges=[('START', test_state_node)]
)

async def main():
    session_service = InMemorySessionService()
    session = await session_service.create_session(user_id="test_user", app_name="test")
    runner = Runner(agent=workflow, session_service=session_service, app_name="test")
    msg = types.Content(role="user", parts=[types.Part.from_text(text="hello")])
    async for event in runner.run_async(new_message=msg, user_id="test_user", session_id=session.id):
        pass
    # Load session again to check final state
    session_after = await session_service.get_session(session.id)
    print("Session state after run:", session_after.state)
    if hasattr(session_after.state, "to_dict"):
        print("State to_dict:", session_after.state.to_dict())
    else:
        print("State attributes:", dir(session_after.state))

asyncio.run(main())
