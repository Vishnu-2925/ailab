import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
data = {
    "message": [
        "Congratulations you won a free lottery",
        "You won a cash prize",
        "Claim your free gift now",
        "You have won a reward",
        "Get free money today",
        "Click this link to win a prize",
        "Win a free mobile phone",
        "Congratulations you are a lucky winner",
        "Free offer available now",
        "You have won a lottery ticket",

        "Hello how are you",
        "Can we meet tomorrow",
        "Please send me the notes",
        "What time is the class",
        "I will call you later",
        "Happy birthday",
        "Are you coming to college",
        "Please call me when you are free",
        "The meeting is at 10 AM",
        "Can you send the assignment"
    ],

    "label": [
        "spam", "spam", "spam", "spam", "spam",
        "spam", "spam", "spam", "spam", "spam",

        "ham", "ham", "ham", "ham", "ham",
        "ham", "ham", "ham", "ham", "ham"
    ]
}

df = pd.DataFrame(data)

messages = df["message"]
labels = df["label"]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)

model = MultinomialNB()
model.fit(X, labels)

print("=" * 50)
print("       AI-BASED SPAM MESSAGE DETECTOR")
print("=" * 50)

message = input("\nEnter your message: ")


message_vector = vectorizer.transform([message])


prediction = model.predict(message_vector)[0]


print("\n" + "=" * 50)

if prediction == "spam":
    print("RESULT : SPAM")
    print("Warning: This message may be spam.")
else:
    print("RESULT : HAM / SAFE")
    print("This message appears to be safe.")

print("=" * 50)
