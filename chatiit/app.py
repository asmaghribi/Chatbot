from flask import Flask, render_template, request, jsonify
from chat import Chat
import joblib

app = Flask(__name__)

# Charger le modèle pré-entrainé
MODEL_FILE = "model.joblib"
model = joblib.load(MODEL_FILE)

# Instancier la classe de chat
chat = Chat()

@app.route("/")
def index_get():
    return render_template("base.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.get_json().get("message")
    response = chat.get_response(text)
    message = {"answer": response}
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)
