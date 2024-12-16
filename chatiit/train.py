import random
import json
import torch
import joblib
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Téléchargement des ressources nécessaires pour le prétraitement
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

class Preprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, sentence):
        # Tokenization
        tokens = word_tokenize(sentence.lower())

        # Suppression de la ponctuation
        tokens = [token for token in tokens if token not in string.punctuation]

        # Suppression des mots vides
        tokens = [token for token in tokens if token not in self.stop_words]

        # Lématisation
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        preprocessed_sentence = ' '.join(tokens)
        return preprocessed_sentence


class Train:
    def __init__(self):
        self.preprocessor = Preprocessor()

    def train_model(self):
        with open('intents.json', 'r', encoding='utf-8') as file:
            data = json.load(file)

        train_x = []
        train_y = []

        for intent in data['intents']:
            for pattern in intent['patterns']:
                preprocessed_pattern = self.preprocessor.preprocess(pattern)
                train_x.append(preprocessed_pattern)
                train_y.append(intent['tag'])

        vectorizer = TfidfVectorizer()
        train_vectors = vectorizer.fit_transform(train_x)

        label_encoder = LabelEncoder()
        train_labels = label_encoder.fit_transform(train_y)

        X_train, X_test, y_train, y_test = train_test_split(train_vectors, train_labels, test_size=0.2, random_state=42)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

        model = GradientBoostingClassifier()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)

        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.1, 0.5, 1],
            'max_depth': [3, 5, 7]
        }
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=2)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print("Best Parameters:", best_params)

        final_model = GradientBoostingClassifier(**best_params)
        final_model.fit(train_vectors, train_labels)

        joblib.dump(final_model, 'model.joblib')
        joblib.dump(vectorizer, 'vectorizer.joblib')
        joblib.dump(label_encoder, 'label_encoder.joblib')
