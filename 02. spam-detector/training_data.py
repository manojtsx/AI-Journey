from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from data_preparation import X, y

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = MultinomialNB()

# Train the model
model.fit(X_train, y_train)

# Print confirmation
print("Model trained successfully!")
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)