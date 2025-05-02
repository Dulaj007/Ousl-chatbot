# Step 1: Import necessary libraries
import random
import nltk
from nltk.tokenize import TreebankWordTokenizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
import numpy as np

# Step 2: Download NLTK resources (only needed once)

# nltk.download('punkt')
# hinltk.download('stopwords')

# Step 3: Define training data
training_data = [
    {"intent": "greeting", "patterns": ["Hi", "Hello", "Hey", "Good morning", "What's up"], "response": "Hello there! 👋"},
    {"intent": "goodbye", "patterns": ["Bye", "See you", "Goodbye", "Later"], "response": "Goodbye! Take care 😊"},
    {"intent": "thanks", "patterns": ["Thanks", "Thank you", "Appreciate it", "Cheers"], "response": "You're welcome! 🙌"},
    {"intent": "unknown", "patterns": [], "response": "Sorry, I didn’t understand that. 🤔"},
]
tokenizer = TreebankWordTokenizer()
# Step 4: Preprocessing
def preprocess(text):
    words = tokenizer.tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    return " ".join([word for word in words if word.isalnum() and word not in stop_words])
# Build training sets
corpus = []
labels = []

for item in training_data:
    for pattern in item["patterns"]:
        corpus.append(preprocess(pattern))
        labels.append(item["intent"])

# Step 5: Vectorization and Model Training
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
model = LogisticRegression()
model.fit(X, labels)

# Step 6: Chat function
def chatbot_response(user_input):
    processed = preprocess(user_input)
    vectorized = vectorizer.transform([processed])
    probabilities = model.predict_proba(vectorized)[0]
    max_prob = np.max(probabilities)
    prediction = model.classes_[np.argmax(probabilities)]

    # Confidence threshold
    if max_prob < 0.5:
        return "Sorry, I didn’t understand that. 🤔"

    # Get matching response
    for item in training_data:
        if item["intent"] == prediction:
            return item["response"]
    return "Sorry, I didn’t understand that. 🤔"

# Step 7: Simple chat loop
print("🤖 Chatbot: Hi! I’m your assistant. Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("🤖 Chatbot: Goodbye!")
        break
    response = chatbot_response(user_input)
    print(f"🤖 Chatbot: {response}")
