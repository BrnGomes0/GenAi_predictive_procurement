from flask import Flask
from routes import ollama

app_flask = Flask(__name__)
app_flask.register_blueprint(blueprint=ollama)
app_flask.run(debug=True)