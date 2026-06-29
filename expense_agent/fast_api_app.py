# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import logging
import base64
import json
import uuid
import google.auth
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Any
from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.cli.fast_api import get_fast_api_app
from expense_agent.agent import root_agent
from expense_agent.app_utils.telemetry import setup_telemetry
from expense_agent.app_utils.typing import Feedback

class PubSubMessage(BaseModel):
    data: Optional[str] = Field(default=None, description="Base64-encoded message data.")
    attributes: Optional[dict[str, str]] = Field(default=None, description="Message attributes.")
    messageId: Optional[str] = Field(default=None, description="Pub/Sub message ID.")
    publishTime: Optional[str] = Field(default=None, description="Publish timestamp.")

class PubSubPushRequest(BaseModel):
    message: PubSubMessage
    subscription: Optional[str] = Field(
        default=None,
        description="Full subscription name (e.g. projects/p/subscriptions/s)."
    )

setup_telemetry()

# Configure standard Python logging for console logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)

class ConsoleStructLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def log_struct(self, data: dict, severity: str = "INFO"):
        level = getattr(logging, severity.upper(), logging.INFO)
        self.logger.log(level, f"Struct Log: {data}")

    def info(self, msg: str, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

logger = ConsoleStructLogger(__name__)

try:
    _, project_id = google.auth.default()
except Exception:
    project_id = None

allow_origins = (
    os.getenv("ALLOW_ORIGINS", "").split(",") if os.getenv("ALLOW_ORIGINS") else None
)

# Artifact bucket for ADK (created by Terraform, passed via env var)
logs_bucket_name = os.environ.get("LOGS_BUCKET_NAME")

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# In-memory session configuration - no persistent storage
session_service_uri = None

artifact_service_uri = f"gs://{logs_bucket_name}" if logs_bucket_name else None

app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=True,
    artifact_service_uri=artifact_service_uri,
    allow_origins=allow_origins,
    session_service_uri=session_service_uri,
    otel_to_cloud=False,
)
app.title = "ambient-expense-agent"
# Initialize Runner for handling Pub/Sub messages
session_service = InMemorySessionService()
runner = Runner(agent=root_agent, session_service=session_service, app_name="expense_agent")


@app.post("/pubsub")
async def handle_pubsub_trigger(request: PubSubPushRequest):
    subscription = request.subscription or "projects/unknown/subscriptions/default-subscription"
    # Normalize subscription path to short name
    short_sub_name = subscription.split("/")[-1]

    if not request.message.data:
        raise HTTPException(status_code=400, detail="Missing message data")

    try:
        decoded_bytes = base64.b64decode(request.message.data)
        decoded_str = decoded_bytes.decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to decode message data: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid base64 encoding: {e}")

    logger.info(f"Received Pub/Sub event for subscription: {short_sub_name} (messageId: {request.message.messageId})")

    user_id = short_sub_name
    session_id = request.message.messageId or str(uuid.uuid4())

    session = await session_service.create_session(
        app_name="expense_agent",
        user_id=user_id,
        session_id=session_id
    )

    new_message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=decoded_str)]
    )

    events = []
    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=new_message
        ):
            events.append(event)
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        logger.info(f"[Workflow Output] {part.text}")
    except Exception as e:
        logger.error(f"Error during workflow execution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Workflow run failed: {str(e)}")

    return {"status": "success", "session_id": session.id, "events_count": len(events)}


@app.post("/feedback")
def collect_feedback(feedback: Feedback) -> dict[str, str]:
    """Collect and log feedback.

    Args:
        feedback: The feedback data to log

    Returns:
        Success message
    """
    logger.log_struct(feedback.model_dump(), severity="INFO")
    return {"status": "success"}


# Main execution
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
