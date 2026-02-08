import requests
import json

API_URL = "http://127.0.0.1:8000/chat"

session_id = None

print("\n🛒 Agentic Commerce Chatbot")
print("Type 'exit' to quit.\n")

while True:
    user_msg = input("You: ")

    if user_msg.lower() == "exit":
        break

    payload = {
        "session_id": session_id,
        "message": user_msg
    }

    try:
        res = requests.post(API_URL, json=payload)
        data = res.json()
    except Exception as e:
        print("Server error:", e)
        continue

    session_id = data["session_id"]

    print("\nBot:", data["reply"])

    # Final JSON output
    if data.get("json_output"):
        print("\n✅ FINAL SHOPPING SPEC")
        print(json.dumps(data["json_output"], indent=2))
        print("\nConversation complete.\n")
        break
