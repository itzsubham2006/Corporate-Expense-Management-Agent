import os
import json
import sys
from dotenv import load_dotenv
load_dotenv()

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from expense_agent.agent import root_agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

def run_and_print():
    session_service = InMemorySessionService()
    runner = Runner(agent=root_agent, session_service=session_service, app_name="expense_agent")
    
    user_id = "test_user"
    session_id = "test_session_1"
    
    session = session_service.create_session_sync(user_id=user_id, app_name="expense_agent")
    
    expense_data = {
        "amount": 250.0,
        "submitter": "Bob",
        "category": "Travel",
        "description": "Hotel stay for developer conference."
    }
    
    prompt_content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=json.dumps(expense_data))]
    )
    
    events = list(runner.run(
        new_message=prompt_content,
        user_id=user_id,
        session_id=session.id
    ))
    
    for i, ev in enumerate(events):
        print(f"Event {i}: {type(ev)}")
        print(json.dumps(ev.model_dump(), indent=2, default=str))

if __name__ == "__main__":
    run_and_print()
