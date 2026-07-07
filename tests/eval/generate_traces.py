import os
import json
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Force stdout/stderr to use UTF-8 on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from expense_agent.agent import root_agent, INJECTION_KEYWORDS
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from vertexai import types as vertex_types

def clean_none_from_dict(d):
    """Recursively remove None values from a dictionary."""
    if not isinstance(d, dict):
        return d
    return {k: clean_none_from_dict(v) for k, v in d.items() if v is not None}

def run_evaluation():
    project_root = Path(__file__).resolve().parent.parent.parent
    dataset_path = project_root / "tests" / "eval" / "datasets" / "basic-dataset.json"
    output_path = project_root / "artifacts" / "traces" / "generated_traces.json"
    
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    eval_cases_input = dataset.get("eval_cases", [])
    print(f"Loaded {len(eval_cases_input)} evaluation cases.")
    
    session_service = InMemorySessionService()
    runner = Runner(agent=root_agent, session_service=session_service, app_name="expense_agent")
    
    eval_cases_output = []
    
    for i, case in enumerate(eval_cases_input):
        case_id = case.get("eval_case_id", f"case_{i}")
        print(f"\n--- Running case {i+1}/{len(eval_cases_input)}: {case_id} ---")
        
        prompt_dict = case.get("prompt")
        # Reconstruct prompt types.Content
        prompt_content = types.Content(
            role=prompt_dict.get("role", "user"),
            parts=[types.Part.from_text(text=p.get("text")) for p in prompt_dict.get("parts", [])]
        )
        
        # Get expense data to check for prompt injection
        try:
            expense_data = json.loads(prompt_content.parts[0].text)
        except Exception:
            expense_data = {"description": prompt_content.parts[0].text}
            
        session = session_service.create_session_sync(user_id=case_id, app_name="expense_agent")
        
        # 1. Run the initial step
        events = list(runner.run(
            new_message=prompt_content,
            user_id=case_id,
            session_id=session.id
        ))
        
        agent_events = []
        # Add the initial prompt as the first event
        agent_events.append({
            "author": "user",
            "content": prompt_content.model_dump()
        })
        
        for ev in events:
            agent_events.append({
                "author": ev.author or "root_agent",
                "content": ev.content.model_dump() if ev.content else None,
                "event_time": datetime.fromtimestamp(ev.timestamp).isoformat() if ev.timestamp else None,
                "state_delta": ev.actions.state_delta if ev.actions and ev.actions.state_delta else None
            })
            
        # 2. Check if we hit HITL
        is_interrupted = False
        for ev in events:
            if ev.content and ev.content.parts:
                for part in ev.content.parts:
                    if part.function_call and part.function_call.name == "adk_request_input":
                        is_interrupted = True
                        break
                        
        if is_interrupted:
            description = expense_data.get("description", "").lower()
            # If description contains injection keywords, reject; otherwise, approve
            has_injection = any(kw in description for kw in INJECTION_KEYWORDS)
            decision = "reject" if has_injection else "approve"
            print(f"Interrupted by human review. Automating decision: {decision}")
            
            resume_message = types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            name="human_review",
                            id="decision",
                            response={"decision": decision}
                        )
                    )
                ]
            )
            
            agent_events.append({
                "author": "user",
                "content": resume_message.model_dump()
            })
            
            # Run the resume step
            resume_events = list(runner.run(
                new_message=resume_message,
                user_id=case_id,
                session_id=session.id
            ))
            
            for ev in resume_events:
                agent_events.append({
                    "author": ev.author or "root_agent",
                    "content": ev.content.model_dump() if ev.content else None,
                    "event_time": datetime.fromtimestamp(ev.timestamp).isoformat() if ev.timestamp else None,
                    "state_delta": ev.actions.state_delta if ev.actions and ev.actions.state_delta else None
                })
                
        # 3. Find final model response
        final_response_text = None
        for ev in reversed(agent_events):
            if ev.get("author") != "user" and ev.get("content"):
                parts = ev["content"].get("parts") or []
                texts = [p.get("text") for p in parts if p.get("text")]
                if texts:
                    final_response_text = "".join(texts)
                    break
                    
        responses = []
        if final_response_text:
            responses.append({
                "response": {
                    "role": "model",
                    "parts": [{"text": final_response_text}]
                }
            })
            
        case_dict = {
            "eval_case_id": case_id,
            "prompt": prompt_content.model_dump(),
            "agent_data": {
                "turns": [
                    {
                        "turn_index": 0,
                        "turn_id": "turn_0",
                        "events": agent_events
                    }
                ]
            },
            "responses": responses
        }
        
        # Clean None values recursively
        case_dict = clean_none_from_dict(case_dict)
        eval_cases_output.append(case_dict)
        print(f"Case {case_id} completed.")

    # Validate and serialize
    dataset_dict = {"eval_cases": eval_cases_output}
    validated_dataset = vertex_types.EvaluationDataset.model_validate(dataset_dict)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(validated_dataset.model_dump_json(indent=2, exclude_none=True))
        
    print(f"\nTraces successfully saved to: {output_path}")

if __name__ == "__main__":
    run_evaluation()
