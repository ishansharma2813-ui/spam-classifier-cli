from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def train_model():
    texts = [
        "Win money now",
        "Hello how are you",
        "Claim your prize",
        "Let's meet tomorrow",
        "Free recharge offer",
        "Are you coming today"
    ]

    labels = ["spam", "ham", "spam", "ham", "spam", "ham"]

    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(texts)

    model = MultinomialNB()
    model.fit(X_vec, labels)

    return model, vectorizer

def predict(text, model, vectorizer):
    text_vec = vectorizer.transform([text])
    return model.predict(text_vec)[0]
    
