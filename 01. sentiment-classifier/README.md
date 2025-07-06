# Sentiment Classifier

A machine learning project that classifies movie reviews as positive or negative using logistic regression and natural language processing techniques.

## Project Overview

This sentiment classifier analyzes movie reviews from the IMDB dataset and predicts whether the sentiment is positive or negative. The project uses a bag-of-words approach with CountVectorizer for text feature extraction and Logistic Regression for classification.

## Dataset

- **Source**: IMDB Dataset.csv
- **Content**: Movie reviews with corresponding sentiment labels
- **Size**: Large dataset (>50MB)
- **Format**: CSV with 'review' and 'sentiment' columns

## Project Structure

```
sentiment-classifier/
├── IMDB Dataset.csv          # Raw dataset
├── data_preparation.py       # Data preprocessing and feature extraction
├── model_training.py         # Model training and saving
├── accuracy_test.py          # Model evaluation and testing
├── app.py                   # Interactive application for predictions
├── requirements.txt         # Python dependencies
├── trained_model.pkl        # Saved trained model
├── vectorizer.pkl           # Saved CountVectorizer
├── X_data.pkl              # Processed feature matrix
├── y_data.pkl              # Target labels
├── X_test.pkl              # Test features
└── y_test.pkl              # Test labels
```

## Features

- **Text Preprocessing**: Converts text to lowercase and removes punctuation
- **Feature Extraction**: Uses CountVectorizer for bag-of-words representation
- **Machine Learning**: Logistic Regression classifier
- **Model Persistence**: Saves trained models using pickle
- **Interactive Interface**: Command-line interface for real-time predictions
- **Model Evaluation**: Comprehensive accuracy testing with classification reports

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dependencies

The project requires the following Python packages:

- joblib==1.5.1
- numpy==2.3.1
- pandas==2.3.0
- python-dateutil==2.9.0.post0
- pytz==2025.2
- scikit-learn==1.7.0
- scipy==1.16.0
- six==1.17.0
- threadpoolctl==3.6.0
- tzdata==2025.2

## Usage

### 1. Data Preparation

Run the data preparation script to process the raw dataset:

```bash
python data_preparation.py
```

This script:
- Loads the IMDB dataset from CSV
- Cleans the text data (lowercase, remove punctuation)
- Converts text to numerical features using CountVectorizer
- Saves processed data as pickle files

### 2. Model Training

Train the logistic regression model:

```bash
python model_training.py
```

This script:
- Loads the processed data
- Splits data into training (80%) and testing (20%) sets
- Trains a Logistic Regression model
- Saves the trained model and test data

### 3. Model Evaluation

Evaluate the model's performance:

```bash
python accuracy_test.py
```

This script provides:
- Test accuracy score
- Detailed classification report
- Confusion matrix
- Actual vs. predicted comparisons

### 4. Interactive Application

Run the interactive sentiment classifier:

```bash
python app.py
```

Features:
- Interactive command-line interface
- Real-time sentiment prediction for user input
- Example predictions on sample reviews
- Type 'exit' to quit the application

## Model Details

- **Algorithm**: Logistic Regression
- **Feature Extraction**: CountVectorizer (Bag-of-Words)
- **Text Preprocessing**: Lowercase conversion and punctuation removal
- **Train/Test Split**: 80% training, 20% testing
- **Random State**: 42 (for reproducibility)

## Code Structure

### data_preparation.py
- Loads and validates the IMDB dataset
- Implements text cleaning function
- Creates feature matrix using CountVectorizer
- Exports processed data for training

### model_training.py
- Imports preprocessed data
- Splits data into train/test sets
- Trains Logistic Regression model
- Saves trained model and test data

### accuracy_test.py
- Evaluates model performance on test data
- Generates comprehensive evaluation metrics
- Displays confusion matrix and classification report

### app.py
- Loads trained model and vectorizer
- Provides interactive prediction interface
- Includes sample predictions for demonstration

## Sample Usage

```python
# Example predictions
sample_reviews = [
    "This movie was fantastic and thrilling!",    # Predicted: positive
    "I didn't enjoy the plot, it was confusing."  # Predicted: negative
]
```

## File Outputs

The project generates several pickle files for model persistence:
- `trained_model.pkl`: Trained Logistic Regression model
- `vectorizer.pkl`: Fitted CountVectorizer
- `X_data.pkl`, `y_data.pkl`: Full processed dataset
- `X_test.pkl`, `y_test.pkl`: Test set for evaluation

## Future Enhancements

- Implement more advanced text preprocessing (stemming, stop word removal)
- Experiment with different algorithms (SVM, Random Forest, Neural Networks)
- Add cross-validation for better model evaluation
- Implement a web interface using Flask or Streamlit
- Add support for multi-class sentiment classification
- Include confidence scores in predictions

## Contributing

Feel free to fork this project and submit pull requests for improvements or bug fixes.

## License

This project is open source and available under the MIT License.
