import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    return text

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    words = nltk.word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(word) for word in words])

def preprocess_data(df):
    df['text'] = df['text'].apply(clean_text).apply(lemmatize_text)
    df['target'] = df['target'].map({0: -1, 2: 0, 4: 1})
    return df

def extract_features(df, method='tfidf', ngram_range=(1, 2), glove_path=None):
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=ngram_range)
        X = vectorizer.fit_transform(df['text']).toarray()
    elif method == 'glove':
        embeddings_index = {}
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        embedding_dim = len(next(iter(embeddings_index.values())))
        
        def embed_text(text):
            words = text.split()
            embedding = np.mean([embeddings_index.get(w, np.zeros(embedding_dim)) for w in words], axis=0)
            return embedding
        
        X = np.array([embed_text(text) for text in df['text']])
    else:
        raise ValueError("Method not supported: choose 'tfidf' or 'glove'")
    return X
