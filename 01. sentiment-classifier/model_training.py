#Choosing and Training the Model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Import X and y from data preparation
with open('X_data.pkl', 'rb') as f:
    X = pickle.load(f)

with open('y_data.pkl', 'rb') as f:
    y = pickle.load(f)

print("Data imported successfully!")
print("X shape:", X.shape)
print("y shape:", y.shape)

#Split the data into training and testing sets (80%train and 20% set)
X_train, X_test, y_train, y_test  = train_test_split(X,y, test_size=0.2, random_state=42)

# Initialize the model
model = LogisticRegression()

#Train the model 
model.fit(X_train, y_train)

#Print confirmation
print("Model Trained Successfully")
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Export trained model and test data for use in accuracy testing
with open('trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('X_test.pkl', 'wb') as f:
    pickle.dump(X_test, f)

with open('y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)

print("Model and test data exported successfully!")