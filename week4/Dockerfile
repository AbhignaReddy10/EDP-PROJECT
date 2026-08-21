FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predict.py .
COPY spam_model.pkl .
COPY tfidf_vectorizer.pkl .

CMD ["python", "predict.py"]