# Week 3 — Spam Classifier (Text Preprocessing, TF-IDF, Naive Bayes)

Built a spam/ham text classifier on the SMS Spam Collection dataset
(5,169 real text messages after removing duplicates).

## Pipeline
1. Loaded tab-separated dataset, inspected for duplicates/missing values
2. Removed 403 duplicate messages
3. Preprocessed text: lowercased, stripped punctuation
4. Train/test split (80/20)
5. Converted text to numeric features using TF-IDF
6. Trained a Multinomial Naive Bayes classifier
7. Evaluated with accuracy, precision, recall, confusion matrix

## Results
- Accuracy: 95.4%
- Spam precision: 1.00 (no real messages ever wrongly flagged as spam)
- Spam recall: 0.66 (catches about 2/3 of actual spam messages)

## Key finding
The model strongly favors avoiding false positives (never blocking a
real message) over catching every spam message — a reasonable tradeoff
for a real-world spam filter, but a good example of why accuracy alone
can be misleading on imbalanced datasets (spam was the minority class).