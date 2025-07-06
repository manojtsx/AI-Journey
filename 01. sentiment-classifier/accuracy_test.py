#Step 4 :  Evaluating the model
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Import trained model and test data from model training
with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)

with open('y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

print("Model and test data imported successfully!")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

#Print results
print("Test Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Optional: Print actual vs. predicted sentiments
print("\nActual vs. Predicted:")
for review, actual, predicted in zip(X_test, y_test, y_pred):
    print(f"Review (numerical): {review.toarray()}, Actual: {actual}, Predicted: {predicted}")

from sklearn.metrics import confusion_matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))     