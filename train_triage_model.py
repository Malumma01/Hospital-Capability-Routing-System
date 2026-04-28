# train_triage_model.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib


# Training dataset (expand this over time)
data = [
    ("my friend was bitten by a snake and is weak", "snakebite"),
    ("snakebite with swelling and pain", "snakebite"),
    ("child has a snakebite and is vomiting", "snakebite"),
    ("venomous snakebite with severe pain", "snakebite"),
    ("cobra bite with swelling and difficulty breathing", "snakebite"),
    ("my dog was bitten by a snake", "snakebite"),
    ("snake attack with swelling and pain", "snakebite"),
    ("a snake bit my brother in the farm", "snakebite"),
    ("snakebite emergency", "snakebite"),
    ("bitten by poisonous snake", "snakebite"),
    ("snakebite victim needs antivenom", "snakebite"),
    ("my child is not breathing and is turning blue", "child_not_breathing"),
    ("baby is not breathing properly", "child_not_breathing"),
    ("child is choking and cannot breathe", "child_not_breathing"),
    ("my child is having trouble breathing", "child_not_breathing"),
    ("my baby is gasping for air and not breathing", "child_not_breathing"),
    ("he is having convulsions and shaking", "seizure"),
    ("she is having a seizure and foaming", "seizure"),
    ("my friend is having a seizure and is unresponsive", "seizure"),
    ("he is convulsing and cannot be woken up", "seizure"),
    ("she is having a seizure and is shaking uncontrollably", "seizure"),
    ("jerking movements and loss of consciousness", "seizure"),
    ("convulsing and unresponsive", "seizure"),
    ("sudden convulsions and shaking", "seizure"),
    ("epileptic seizure with convulsions", "seizure"),
    ("convulsions due to seizure", "seizure"),
    ("he has mild fever and headache", "other"),
    ("stomach pain since yesterday", "other"),
    ("mild headache and nausea", "other"),
    ("back pain and fatigue", "other"),
    ("sore throat and cough", "other"),
    ("minor cut with bleeding", "other"),
    ("sprained ankle", "other"),
    ("general weakness and dizziness", "other"),
]

df = pd.DataFrame(data, columns=["text", "label"])

vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1)
X = vectorizer.fit_transform(df["text"])
y = df["label"]

clf = LogisticRegression()
clf.fit(X, y)

joblib.dump(vectorizer, "triage_vectorizer.joblib")
joblib.dump(clf, "triage_model.joblib")

print("Model training complete. Files saved:")
print(" - triage_vectorizer.joblib")
print(" - triage_model.joblib")