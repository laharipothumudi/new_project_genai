"""
Flask service — SHL Assessment Recommender
Endpoints:
  GET  /health  → {"status": "ok"}
  POST /chat    → ChatResponse
"""
import os
from flask import Flask, request, jsonify
import agent
import vector_store

app = Flask(__name__)

# ── Warm up vector store on startup ──────────────────────────────────────────
with app.app_context():
    if os.path.exists(vector_store.INDEX_PATH):
        vector_store.search("warmup", k=1)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _validate_messages(messages):
    if not isinstance(messages, list) or len(messages) == 0:
        return "messages must be a non-empty list"
    for m in messages:
        if not isinstance(m, dict):
            return "each message must be an object"
        if m.get("role") not in ("user", "assistant"):
            return f"invalid role '{m.get('role')}': must be 'user' or 'assistant'"
        if not isinstance(m.get("content"), str):
            return "each message must have a string 'content' field"
    return None


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.post("/chat")
def chat():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "request body must be JSON"}), 400

    err = _validate_messages(body.get("messages"))
    if err:
        return jsonify({"error": err}), 400

    messages = [{"role": m["role"], "content": m["content"]} for m in body["messages"]]

    try:
        result = agent.run(messages)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    recs = result.get("recommendations", [])
    if not isinstance(recs, list):
        recs = []

    return jsonify({
        "reply": result.get("reply", ""),
        "recommendations": [
            {
                "name": r.get("name", ""),
                "url": r.get("url", ""),
                "test_type": r.get("test_type", ""),
            }
            for r in recs[:10]
        ],
        "end_of_conversation": bool(result.get("end_of_conversation", False)),
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
