# Week 4 — MLOps: Package the Spam Classifier with Docker

Packaged the Week 3 spam classifier (TF-IDF + Naive Bayes) into a
Docker container for portable, environment-independent deployment.

## Files
- `predict.py` — standalone script that loads the saved model/vectorizer and makes predictions
- `spam_model.pkl`, `tfidf_vectorizer.pkl` — trained model artifacts (saved with joblib)
- `requirements.txt` — Python dependencies
- `Dockerfile` — build instructions for the container image

## How to run
docker build -t spam-classifier .
docker run spam-classifier

## What I learned
- Images vs containers vs Dockerfiles
- Why train/inference code should be separated from the trained artifacts
- How to package a model so it runs identically on any machine