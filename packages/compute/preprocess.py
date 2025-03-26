import ast
import re
from typing import List

import nltk
import pandas as pd

# download preprocessing assets (corpus and word lists)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


def clean(dataset_path: str) -> str:
    """
    Applies basic text cleaning to the 'text' column.
    - Converts to lowercase
    - Removes URLs
    - Removes special characters
    - Removes extra whitespace

    Parameters
    ----------
    dataset_path: `str`
    The dataset CSV path containing tweets with sentiment labels.

    Returns
    -------
    `str` The path for the clean version of the dataset.
    """
    def _clean_text(text: str):
        # Convert to lowercase
        text = text.lower().strip()
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    dtypes = {
        'ids': int,
        'date': str,
        'flag': str,
        'user': str,
        'text': str,
        'target': int  # 0=negative, 2=neutral, 4=positive
    }

    dataset_path = f"{dataset_path}/dataset.csv"
    new_path = "/result/dataset.csv"
    
    # Read and clean the dataset
    df = pd.read_csv(dataset_path, dtype=dtypes)
    df["text"] = df["text"].apply(_clean_text)
    df.to_csv(new_path, index=False)
    
    return new_path


def tokenize(dataset_path: str) -> str:
    """
    Creates a 'tokens' column with lemmatized tokens.
    This helps in standardizing words for better sentiment analysis.

    Parameters
    ----------
    dataset_path: `str`
    The dataset CSV path containing cleaned tweets.

    Returns
    -------
    `str` The path for the tokenized version of the dataset.
    """
    dtypes = {
        'ids': int,
        'date': str,
        'flag': str,
        'user': str,
        'text': str,
        'target': int  # only in training data
    }

    dataset_path = f"{dataset_path}/dataset.csv"
    new_path = "/result/dataset.csv"
    
    # Read the dataset
    df = pd.read_csv(dataset_path, dtype=dtypes)
    
    # Create tokens and lemmatize
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    lemmatizer = nltk.stem.WordNetLemmatizer()
    
    def _tokenize_and_lemmatize(text: str) -> List[str]:
        tokens = tokenizer.tokenize(text)
        return [lemmatizer.lemmatize(token) for token in tokens]
    
    df["tokens"] = df["text"].apply(_tokenize_and_lemmatize)
    df.to_csv(new_path, index=False)
    
    return new_path


def remove_stopwords(dataset_path: str) -> str:
    """
    Removes common English stopwords from the tokens.
    Note: Some stopwords might be important for sentiment,
    so we keep certain ones that might indicate sentiment.

    Parameters
    ----------
    dataset_path: `str`
    The dataset CSV path with tokenized tweets.

    Returns
    -------
    `str` The path for the filtered version of the dataset.
    """
    dtypes = {
        'ids': int,
        'date': str,
        'flag': str,
        'user': str,
        'text': str,
        'target': int
    }

    dataset_path = f"{dataset_path}/dataset.csv"
    new_path = "/result/dataset.csv"
    
    # Get stopwords but keep some sentiment-relevant ones
    stopwords = set(nltk.corpus.stopwords.words('english'))
    # Keep negative words that might be important for sentiment
    sentiment_words = {'no', 'not', 'nor', 'none', 'never', 'nothing'}
    stopwords = stopwords - sentiment_words

    def _remove_stopwords(tokens: List[str]) -> List[str]:
        return [token for token in tokens if token not in stopwords]

    # Read and process the dataset
    df = pd.read_csv(
        dataset_path,
        dtype=dtypes,
        converters={"tokens": ast.literal_eval}
    )
    
    df["tokens"] = df["tokens"].apply(_remove_stopwords)
    df.to_csv(new_path, index=False)
    
    return new_path