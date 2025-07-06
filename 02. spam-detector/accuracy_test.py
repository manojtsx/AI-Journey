from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from training_data import model, X_test, y_test

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print results
print("Test Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Optional: Print actual vs. predicted labels
print("\nActual vs. Predicted:")
for msg, actual, predicted in zip(X_test, y_test, y_pred):
    print(f"Message (numerical): {msg.toarray()}, Actual: {actual}, Predicted: {predicted}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))