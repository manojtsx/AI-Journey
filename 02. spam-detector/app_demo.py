#!/usr/bin/env python3
"""
Non-interactive spam detector app for demonstration
"""
import joblib
from training_data import model
from data_preparation import vectorizer, clean_text

# Save the model and vectorizer for future use
joblib.dump(model, 'spam_detector_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("✓ Model and vectorizer saved to disk")

# Function to predict label for a new message
def predict_spam(message, vectorizer, model):
    # Clean the input message
    message = clean_text(message)
    # Convert to numerical features
    message_vector = vectorizer.transform([message])
    # Predict label
    prediction = model.predict(message_vector)
    return prediction[0]

# Demo with example messages
print("\n=== Spam Detector Demo ===")
sample_messages = [
    "Win a free trip today! Click now!",
    "Let's schedule a call for tomorrow.",
    "URGENT: Claim your prize now!!!",
    "Hi, how are you doing?",
    "Get rich quick with this amazing offer!"
]

for message in sample_messages:
    label = predict_spam(message, vectorizer, model)
    print(f"Message: '{message}'")
    print(f"Prediction: {label.upper()}")
    print("-" * 50)

print("\nTo use the interactive version, run: python app.py")
