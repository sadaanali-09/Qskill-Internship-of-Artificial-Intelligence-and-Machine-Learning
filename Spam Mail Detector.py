import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

print("--- SMS Spam Collection Sample ---")
print(df.head())
print("\nClass distribution:\n", df['label'].value_counts())

df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, Y_train, y_test = train_test_split(df['message'], df['label_num'], test_size=0.2, random_state=42, stratify=df['label_num'])

vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, Y_train)

y_pred = model.predict(X_test_tfidf)

print("\n--- Task 2 Evaluation Results ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

user_message = input("\nEnter a message to check: ").strip()
if not user_message:
    print("No message was entered.")
else:
    user_message_tfidf = vectorizer.transform([user_message])
    prediction = model.predict(user_message_tfidf)[0]
    result = "Spam" if prediction == 1 else "Ham (not spam)"
    print(f"\nPrediction: {result}")