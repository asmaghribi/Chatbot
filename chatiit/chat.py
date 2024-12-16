import joblib
import random
import json
from preprocessing import Preprocessor

class Chat:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.model = joblib.load("model.joblib")
        self.vectorizer = joblib.load("vectorizer.joblib")
        self.label_encoder = joblib.load("label_encoder.joblib")
        self.data = self.load_data()

    def load_data(self):
        with open('intents.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data

    def get_response(self, msg):
        sentence = self.preprocessor.preprocess(msg)
        X = self.vectorizer.transform([sentence])

        output = self.model.predict(X)
        tag = self.label_encoder.inverse_transform(output)[0]

        for intent in self.data['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

        return "I do not understand..."

