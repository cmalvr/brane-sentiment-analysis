import ast
import datetime
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

def train_model(dataset_path: str) -> str:
    """
    Trains a logistic regression classifier for sentiment analysis.
    The trained model is stored as a binary file in the DFS.

    Parameters
    ----------
    dataset_path: `str`
    CSV file containing the preprocessed train dataset with sentiment labels.

    Returns
    -------
    `str` The path for the trained model binary dump in the DFS.
    """
    dtypes = {
        'ids': int,
        'date': str,
        'flag': str,
        'user': str,
        'text': str,
        'target': int,  # 0 = negative, 2 = neutral, 4 = positive
    }

    # Load the dataset
    dataset = pd.read_csv(
        dataset_path,
        index_col='ids',
        dtype=dtypes,
        converters={"tokens": ast.literal_eval})

    # Convert tokens back to text for vectorization
    dataset['processed_text'] = dataset['tokens'].apply(' '.join)

    # Create vectorizer and transform text
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(dataset['processed_text'])

    # Initialize and train the model
    model = LogisticRegression(
        multi_class='multinomial',
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    
    # Map sentiment values for internal model training
    sentiment_map = {0: 0, 2: 1, 4: 2}  # negative: 0, neutral: 1, positive: 2
    y = dataset["target"].map(sentiment_map)
    
    model.fit(X, y)

    # Save the model, mapping, and vectorizer
    with open("/result/model.pickle", "wb") as f:
        pickle.dump({
            'model': model,
            'vectorizer': vectorizer,
            'sentiment_map': sentiment_map
        }, f)

    return "/result/model.pickle"


def create_submission(dataset_path: str, model_path: str) -> str:
    """
    Performs sentiment analysis on each tweet in the test dataset.
    It stores the final result as a CSV in the DFS in the form:

    `ids,target,negative_conf,neutral_conf,positive_conf`

    Where target matches original values:
    0 = negative
    2 = neutral
    4 = positive

    Parameters
    ----------
    dataset_path: `str`
    CSV file containing the preprocessed test dataset.

    model_path: `str`
    Binary file containing the trained model.

    Returns
    -------
    `str` The path for the submission CSV.
    """
    dtypes = {
        'ids': int,
        'date': str,
        'flag': str,
        'user': str,
        'text': str,
    }

    # Load the test dataset
    dataset = pd.read_csv(
        dataset_path,
        index_col='ids',
        dtype=dtypes,
        converters={"tokens": ast.literal_eval})

    # Convert tokens back to text
    dataset['processed_text'] = dataset['tokens'].apply(' '.join)

    # Load model artifacts
    with open(model_path, "rb") as f:
        artifacts = pickle.load(f)
        model = artifacts['model']
        vectorizer = artifacts['vectorizer']
        sentiment_map = artifacts['sentiment_map']

    # Transform text using the same vectorizer
    X = vectorizer.transform(dataset['processed_text'])

    # Get predictions and probabilities
    numeric_predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    # Map predictions back to original values (0, 2, 4)
    reverse_sentiment_map = {0: 0, 1: 2, 2: 4}  # Map model outputs back to original values
    predictions = [reverse_sentiment_map[pred] for pred in numeric_predictions]

    # Create submission dataframe
    submission = pd.DataFrame()
    submission['ids'] = dataset.index.to_list()
    submission['target'] = predictions
    
    # Add confidence scores
    submission['negative_conf'] = probabilities[:, 0]  # confidence for 0 (negative)
    submission['neutral_conf'] = probabilities[:, 1]   # confidence for 2 (neutral)
    submission['positive_conf'] = probabilities[:, 2]  # confidence for 4 (positive)

    # Save submission file
    filename = f"/result/submission.csv"
    submission.to_csv(filename, index=False)

    return filename