# Step 5: Using the Model in a Project
import pickle
import re

# Import trained model and vectorizer
with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

print("Model and vectorizer loaded successfully!")

# Clean text function (same as in data_preparation.py)
def clean_text(text):
    text = text.lower()  #Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  #Remove puctuation
    return text

# Function to predict sentiment for a new review
def predict_sentiment(review, vectorizer, model):
    # Clean the input review
    review = clean_text(review)  # Using the clean_text function from Step 2
    # Convert to numerical features
    review_vector = vectorizer.transform([review])
    # Predict sentiment
    prediction = model.predict(review_vector)
    return prediction[0]

# Interactive script to get user input
print("Sentiment Classifier: Enter a movie review to predict its sentiment.")
while True:
    user_review = input("Enter your review (or type 'exit' to quit): ")
    if user_review.lower() == 'exit':
        break
    sentiment = predict_sentiment(user_review, vectorizer, model)
    print(f"Predicted Sentiment: {sentiment}\n")

# Example usage
sample_reviews = [
    "This movie was fantastic and thrilling!",
    "I didn’t enjoy the plot, it was confusing."
]
for review in sample_reviews:
    sentiment = predict_sentiment(review, vectorizer, model)
    print(f"Review: {review}\nPredicted Sentiment: {sentiment}\n")