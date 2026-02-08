import os
import uuid
import json
from typing import Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI()

sessions: Dict[str, Dict[str, Any]] = {}

SYSTEM_PROMPT = """
You are an intelligent shopping assistant.

You help users plan purchases conversationally.

Goal:
Collect information and build a shopping plan.

Ask questions when information is missing.

When complete and user confirms,
respond with:

FINAL_JSON:
<json>

JSON format:
{
  "delivery_pincode": "...",
  "delivery_deadline_date": "...",
  "items": [
    {
      "query": "...",
      "max_price": number,
      "preferences": "..."
    }
  ]
}

Do not output FINAL_JSON until confirmed.
Keep conversation natural.
"""

class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    json_output: dict | None = None


def get_session(session_id):
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "messages": []
        }
    return session_id, sessions[session_id]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

    session_id, session = get_session(req.session_id)

    session["messages"].append(
        {"role": "user", "content": req.message}
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += session["messages"]

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.4,
    )

    assistant_reply = response.choices[0].message.content
    session["messages"].append(
        {"role": "assistant", "content": assistant_reply}
    )

    # Detect final JSON
    if "FINAL_JSON:" in assistant_reply:
        json_part = assistant_reply.split("FINAL_JSON:")[1].strip()
        try:
            parsed = json.loads(json_part)
            return ChatResponse(
                session_id=session_id,
                reply="Order specification completed.",
                json_output=parsed,
            )
        except:
            pass

    return ChatResponse(
        session_id=session_id,
        reply=assistant_reply,
    )
