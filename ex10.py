
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Sample training dataset
messages = [
    "Win a free iPhone now",
    "Congratulations you won a lottery",
    "Claim your free prize now",
    "Click here to win cash",
    "Urgent! Claim your reward",
    "You have won a free gift",
    "Get free money now",
    "Limited offer buy now",
    "Meeting is scheduled for tomorrow",
    "Hello how are you",
    "Please call me when you are free",
    "Your class starts at ten",
    "Can we meet today",
    "I will reach home soon",
    "Please send me the notes",
    "Thank you for your help",
    "Are you coming to college",
    "Your assignment is ready",
    "See you tomorrow",
    "Have a nice day"
]

labels = [
    "Spam", "Spam", "Spam", "Spam", "Spam",
    "Spam", "Spam", "Spam",
    "Ham", "Ham", "Ham", "Ham", "Ham",
    "Ham", "Ham", "Ham", "Ham", "Ham",
    "Ham", "Ham"
]

# Create and train the AI model
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

model.fit(messages, labels)

print("===== AI SPAM DETECTION SYSTEM =====")

while True:
    text = input("\nEnter a message (or type exit): ")

    if text.lower() == "exit":
        print("Thank you!")
        break

    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    confidence = max(probabilities) * 100

    print("Prediction:", prediction)
    print("Model confidence: {:.2f}%".format(confidence))

    if prediction == "Spam":
        print("Warning: This message may be spam!")
    else:
        print("This message appears genuine.")
