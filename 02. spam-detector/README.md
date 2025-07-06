# Spam Detector Project

A machine learning project that classifies text messages as spam or ham (not spam) using a real dataset of 5,572 messages.

## Dataset

The project uses `mail_data.csv` containing:
- **Total messages**: 5,572
- **Spam messages**: 747 (13.4%)
- **Ham messages**: 4,825 (86.6%)

## Model Performance

- **Test Accuracy**: 98.2%
- **Ham Precision**: 99%
- **Spam Precision**: 93%
- **Training Features**: 9,542 unique words
- **Training/Test Split**: 80%/20%

## File Structure

- `data_preparation.py` - Prepares the dataset and creates the text vectorizer
- `training_data.py` - Trains the machine learning model 
- `accuracy_test.py` - Evaluates model performance
- `app.py` - Interactive spam detector application
- `app_demo.py` - Non-interactive demo version
- `main.py` - Runs the complete pipeline
- `test_integration.py` - Tests integration between modules

## Variable Dependencies

The files share variables through imports:

### data_preparation.py exports:
- `X` - Feature matrix (vectorized text)
- `y` - Labels (spam/ham)
- `vectorizer` - Text vectorizer object
- `clean_text()` - Text cleaning function

### training_data.py exports:
- `model` - Trained MultinomialNB model
- `X_train`, `X_test` - Training/testing features
- `y_train`, `y_test` - Training/testing labels

### Import Chain:
1. `training_data.py` imports `X, y` from `data_preparation.py`
2. `accuracy_test.py` imports `model, X_test, y_test` from `training_data.py`
3. `app.py` imports `model` from `training_data.py` and `vectorizer, clean_text` from `data_preparation.py`

## Usage

### Run Complete Pipeline:
```bash
python main.py
```

### Run Individual Steps:
```bash
python data_preparation.py
python training_data.py
python accuracy_test.py
```

### Run Applications:
```bash
# Interactive version
python app.py

# Demo version
python app_demo.py
```

### Test Integration:
```bash
python test_integration.py
```

## Generated Files

- `spam_detector_model.pkl` - Saved trained model
- `vectorizer.pkl` - Saved text vectorizer

## Dependencies

- pandas
- scikit-learn
- joblib
- re (built-in)
