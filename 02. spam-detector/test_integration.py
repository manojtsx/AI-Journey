#!/usr/bin/env python3
"""
Integration test script to verify all modules work together
"""
from training_data import model
from data_preparation import vectorizer, clean_text

def predict_spam(message, vectorizer, model):
    # Clean the input message
    message = clean_text(message)
    # Convert to numerical features
    message_vector = vectorizer.transform([message])
    # Predict label
    prediction = model.predict(message_vector)
    return prediction[0]

# Test messages
sample_messages = [
    "Win a free trip today! Click now.",
    "Let's schedule a call for tomorrow."
]

print("Testing spam detector integration:")
for message in sample_messages:
    label = predict_spam(message, vectorizer, model)
    print(f"Message: '{message}' -> Predicted Label: {label}")

print("\nIntegration test completed successfully!")
