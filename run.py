from flask import Flask
from app.routes import ollama

app_flask = Flask(__name__)
app_flask.register_blueprint(blueprint=ollama)
app_flask.run(host="0.0.0.0", port=5000, debug=True)