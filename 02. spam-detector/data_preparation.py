import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re

# Load the real dataset from CSV
df = pd.read_csv('mail_data.csv')

# Rename columns to match our code expectations
df.columns = ['label', 'message']

# Display basic info about the dataset
print(f"Dataset loaded successfully!")
print(f"Total messages: {len(df)}")
print(f"Spam messages: {len(df[df['label'] == 'spam'])}")
print(f"Ham messages: {len(df[df['label'] == 'ham'])}")
print(f"\nFirst few rows:")
print(df.head())

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

df['message'] = df['message'].apply(clean_text)

#Convert text to numerical features (bag-of-words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])  # Features (numerical representation of text)
y = df['label']  # Labels (spam/ham)    