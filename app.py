import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from common locations.
project_dir = Path(__file__).resolve().parent
dotenv_files = [
    project_dir / ".env",
    project_dir / "key.env",
    project_dir.parent / ".env",
    project_dir.parent / "key.env",
]
for dotenv_path in dotenv_files:
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)

# Constants
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY or GOOGLE_API_KEY not found in environment variables. "
        "Create a .env or key.env file containing GEMINI_API_KEY, or set GOOGLE_API_KEY."
    )

MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
SYSTEM_INSTRUCTION = (
    "You are an expert Academic Tutor. "
    "Strict Rule: Only answer questions related to school, university, or general learning. "
    "If a user asks about entertainment, sports, or non-study topics, politely say: "
    "'I am a study assistant. Please ask a question related to your education.'"
)
ERROR_MESSAGE = "I'm having trouble connecting to my brain right now."

# Initialize Flask app
app = Flask(__name__)

# Configure Gemini API

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    system_instruction=SYSTEM_INSTRUCTION
)


@app.route('/')
def home():
    """Serve the home page."""
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    """Handle user questions and return AI-generated responses."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request format"}), 400

    user_input = data.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    try:
        response = model.generate_content(user_input)
        reply_text = getattr(response, "text", None) or getattr(response, "output_text", None)
        if not reply_text:
            raise ValueError("No valid response text returned from the model.")
        return jsonify({"reply": reply_text})
    except Exception as exc:
        app.logger.exception("Gemini request failed")
        return jsonify({"error": ERROR_MESSAGE}), 500


if __name__ == '__main__':
    app.run(debug=True)