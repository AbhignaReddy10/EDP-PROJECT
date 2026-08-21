import joblib
import string

# Load the saved model and vectorizer once, when the script starts
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    return text

def predict_message(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)
    return prediction[0]

if __name__ == "__main__":
    test_messages = [
        "Congratulations! You've won a free iPhone, click here to claim now!!!",
        "Hey, are we still on for lunch tomorrow?"
    ]
    for msg in test_messages:
        result = predict_message(msg)
        print(f"'{msg}' -> {result}")