from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

emails = [
    "Win money now",
    "Claim your free prize",
    "Congratulations you won lottery",
    "Hi how are you",
    "Let's meet tomorrow",
    "Project meeting at 10am"
]

labels = [
    "spam",
    "spam",
    "spam",
    "ham",
    "ham",
    "ham"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

model = MultinomialNB()
model.fit(X, labels)

user_email = input("Enter an email message: ")

X_test = vectorizer.transform([user_email])

prediction = model.predict(X_test)

print("Prediction:", prediction[0])