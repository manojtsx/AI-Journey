import joblib
from training_data import model
from data_preparation import vectorizer, clean_text

# Save the model and vectorizer for future use
joblib.dump(model, 'spam_detector_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Function to predict label for a new message
def predict_spam(message, vectorizer, model):
    # Clean the input message
    message = clean_text(message)  # Using the clean_text function from Step 2
    # Convert to numerical features
    message_vector = vectorizer.transform([message])
    # Predict label
    prediction = model.predict(message_vector)
    return prediction[0]

# Interactive script to get user input
print("Spam Detector: Enter a message to check if it's spam or ham.")
while True:
    user_message = input("Enter your message (or type 'exit' to quit): ")
    if user_message.lower() == 'exit':
        break
    label = predict_spam(user_message, vectorizer, model)
    print(f"Predicted Label: {label}\n")

# Example usage
sample_messages = [
    "Win a free trip today! Click now.",
    "Let’s schedule a call for tomorrow."
]
for message in sample_messages:
    label = predict_spam(message, vectorizer, model)
    print(f"Message: {message}\nPredicted Label: {label}\n")