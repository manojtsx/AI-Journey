import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
import pickle

# Load the IMDB dataset from CSV file
print("Loading IMDB dataset from CSV...")
df = pd.read_csv('IMDB Dataset.csv')

print(f"Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"First few rows:")
print(df.head())

# Check if the expected columns exist
if 'review' not in df.columns or 'sentiment' not in df.columns:
    print("Warning: Expected columns 'review' and 'sentiment' not found.")
    print("Available columns:", df.columns.tolist())
    # Try to identify the correct columns
    review_col = None
    sentiment_col = None
    
    for col in df.columns:
        if 'review' in col.lower() or 'text' in col.lower():
            review_col = col
        elif 'sentiment' in col.lower() or 'label' in col.lower():
            sentiment_col = col
    
    if review_col and sentiment_col:
        print(f"Using columns: '{review_col}' for reviews and '{sentiment_col}' for sentiment")
        df = df.rename(columns={review_col: 'review', sentiment_col: 'sentiment'})
    else:
        print("Could not identify the correct columns. Please check your CSV file.")
        exit(1)

# data = {
#     'review': [
#         'This movie was amazing and I loved it!',
#         'Terrible film, really boring.',
#         'Great acting and wonderful story.',
#         'Awful, I hated this movie.',
#         'Fantastic experience, highly recommend!'
#     ],
#     'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive']
# }


# Clean text data
def clean_text(text):
    text = text.lower()  #Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  #Remove puctuation
    return text

df['review'] = df['review'].apply(clean_text)

#convert text to numerical features (bag-of-words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['review'])
y = df['sentiment']

# Print the shape of the data
# print("feature matrix shape:", X.shape)
# print("Labels:",y)
# print(vectorizer.get_feature_names_out())

# Export X, y, and vectorizer for use in model training and app
with open('X_data.pkl', 'wb') as f:
    pickle.dump(X, f)

with open('y_data.pkl', 'wb') as f:
    pickle.dump(y, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Data exported successfully!")
print("X shape:", X.shape)
print("y shape:", y.shape)


