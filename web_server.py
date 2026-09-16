"""Web server for the RAG dashboard."""

from pathlib import Path
import logging

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from main import setup_pipeline


ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
logger = logging.getLogger(__name__)
_pipeline = None
_config = None


def get_pipeline():
    """Create the configured pipeline once, on the first API request."""
    global _pipeline, _config
    if _pipeline is None:
        _pipeline, _config = setup_pipeline()
    return _pipeline, _config


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/api/status")
def status():
    try:
        pipeline, _ = get_pipeline()
        return jsonify(pipeline.get_status())
    except Exception as error:
        logger.exception("Could not load pipeline status")
        return jsonify({"error": str(error)}), 500


@app.post("/api/ingest")
def ingest():
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"error": "Choose a PDF file first."}), 400
    if not upload.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported."}), 400

    try:
        pipeline, config = get_pipeline()
        filename = secure_filename(upload.filename)
        pdf_path = config.input_dir / filename
        upload.save(pdf_path)
        result = pipeline.ingest_pdf(
            pdf_path=pdf_path,
            force_recreate=request.form.get("recreate") == "true",
        )
        return jsonify(result)
    except Exception as error:
        logger.exception("PDF ingestion failed")
        return jsonify({"error": str(error)}), 500


@app.post("/api/query")
def query():
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("query", "")).strip()
    if not question:
        return jsonify({"error": "Enter a question first."}), 400

    try:
        pipeline, _ = get_pipeline()
        mode = payload.get("retrieval_mode") or payload.get("mode")
        fusion = payload.get("fusion_method") or payload.get("fusion")
        result = pipeline.rag_query(
            query=question,
            n_retrieve=max(1, min(int(payload.get("top_k", 5)), 20)),
            temperature=float(payload.get("temperature", 0.7)),
            max_tokens=max(1, int(payload.get("max_tokens", 2048))),
            mode=mode,
            fusion_method=fusion,
        )
        return jsonify(result)
    except Exception as error:
        logger.exception("RAG query failed")
        return jsonify({"error": str(error)}), 500


@app.post("/api/quiz/generate")
def quiz_generate():
    payload = request.get_json(silent=True) or {}
    num_questions = max(1, min(int(payload.get("num_questions", 5)), 15))

    try:
        pipeline, _ = get_pipeline()
        questions = pipeline.generate_quiz(num_questions=num_questions)
        # Includes "context" and "expected_answer" (the grading key) alongside each
        # question — the frontend keeps these in memory and sends them back
        # unchanged when the answer is submitted for grading, but never displays them.
        return jsonify({"questions": questions})
    except Exception as error:
        logger.exception("Quiz generation failed")
        return jsonify({"error": str(error)}), 500


@app.post("/api/quiz/evaluate")
def quiz_evaluate():
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question", "")).strip()
    expected_answer = str(payload.get("expected_answer", "")).strip()
    context = str(payload.get("context", "")).strip()
    user_answer = str(payload.get("user_answer", "")).strip()

    if not question or not user_answer:
        return jsonify({"error": "Missing question or answer."}), 400

    try:
        pipeline, _ = get_pipeline()
        result = pipeline.evaluate_answer(
            question=question,
            expected_answer=expected_answer,
            context=context,
            user_answer=user_answer,
        )
        return jsonify(result)
    except Exception as error:
        logger.exception("Answer evaluation failed")
        return jsonify({"error": str(error)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
