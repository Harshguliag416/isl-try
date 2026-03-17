import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

from services.classifier import GestureClassifier
from services.db import ConversationStore

load_dotenv()

app = Flask(__name__)
CORS(app)

classifier = GestureClassifier()
conversation_store = ConversationStore(
    mongo_uri=os.getenv("MONGODB_URI", "").strip(),
    database_name=os.getenv("MONGODB_DB", "isl_bridge"),
)


@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model_type": classifier.model_type,
            "labels": classifier.labels,
        }
    )


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    landmarks = payload.get("landmarks", [])
    mode = payload.get("mode", "sign_to_text")

    if not isinstance(landmarks, list) or len(landmarks) != 21:
        return jsonify({"error": "Expected 21 hand landmarks."}), 400

    prediction = classifier.predict(landmarks)
    return jsonify(
        {
            "prediction": prediction["label"],
            "confidence": prediction["confidence"],
            "source": prediction["source"],
            "mode": mode,
        }
    )


@app.get("/api/conversations")
def list_conversations():
    return jsonify({"items": conversation_store.fetch_recent(limit=8)})


@app.post("/api/conversations")
def create_conversation():
    payload = request.get_json(silent=True) or {}
    item = {
        "mode": payload.get("mode", "unknown"),
        "text": payload.get("text", "").strip(),
        "confidence": float(payload.get("confidence", 0) or 0),
        "created_at": payload.get("created_at")
        or datetime.now(timezone.utc).isoformat(),
    }

    if not item["text"]:
        return jsonify({"error": "Text is required."}), 400

    conversation_store.insert(item)
    return jsonify({"ok": True, "item": item}), 201


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
