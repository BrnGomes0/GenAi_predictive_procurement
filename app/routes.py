from flask import Flask, jsonify, request, Blueprint
from app.ollama_service import ServiceOllama

ollama = Blueprint("ollama", __name__, url_prefix="/ollama")

@ollama.route("/test", methods=["GET"])
def test_application():
    return jsonify({
        "message": "The application OLLAMA it's working..."
    })


@ollama.route("/create", methods=["POST"])
def ask_to_ollama():
    data = request.json()
    prompt = data.get("prompt")
    service_ollama = ServiceOllama(prompt=prompt)
    return jsonify({
        "message": service_ollama.response
    })