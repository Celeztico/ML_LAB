'''

7.Implement a Naïve Bayes classifier to categorize text documents into topics using the 20 Newsgroups dataset. Compare the performance of Multinomial Naïve Bayes with Bernoulli Naïve Bayes.
Tasks:
● Load and preprocess the 20 Newsgroups dataset.
● Implement Multinomial Naïve Bayes and Bernoulli Naïve Bayes classifiers.
● Evaluate and compare the performance of both models using metrics such as
accuracy and F1-score.
● Discuss the strengths and weaknesses of each Naïve Bayes variant for text
classification.

'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score

# Load dataset
df = pd.read_csv("datasets/20newsgroups.csv")
X = df["text"].fillna("")
y = df["target"]

# Convert text to features
vectorizer = CountVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train and evaluate both models
for name, model in [
    ("Multinomial NB", MultinomialNB()),
    ("Bernoulli NB", BernoulliNB(binarize=0.0))
]:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("F1-score:", f1_score(y_test, pred, average="weighted"))