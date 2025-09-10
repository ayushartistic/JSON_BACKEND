from flask import Flask, request, jsonify
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List
import os
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment")

client = genai.Client(api_key=API_KEY)

class CareerProfile(BaseModel):
    personality_traits: List[str]
    best_fit_career: str
    alternative_career: str
    short_explanation: str
    academic_courses: List[str]

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json() or {}
        input_text = data.get("text")
        if not input_text:
            return jsonify({"error": "No text provided"}), 400

        # Use the SDK's documented config object
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Convert this text into the CareerProfile schema:\n\n{input_text}",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CareerProfile,   # or list[CareerProfile] if you want an array
            ),
        )

        # Prefer structured parsed object if SDK provides it
        if getattr(response, "parsed", None):
            profile: CareerProfile = response.parsed
            return jsonify(profile.dict())

        # Fallback: try to parse raw JSON text and validate with Pydantic
        parsed_raw = json.loads(response.text)
        # If response was an array and you expect a single object, adjust accordingly
        validated = CareerProfile.parse_obj(parsed_raw)
        return jsonify(validated.dict())

    except Exception as e:
        return jsonify({"error": "processing_failed", "detail": str(e), "raw": getattr(response, "text", None)}), 500

if __name__ == "__main__":
    app.run(debug=True)
