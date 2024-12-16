import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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
