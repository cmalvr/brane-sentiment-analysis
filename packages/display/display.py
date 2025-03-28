import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

def generate_cs_plot(filepath_sub_dataset: str) -> str:
    """
    Generate a plot that the top N counts of the dataset
    and exports an image in the DFS.

    Parameters
    ----------
    filepath_test_dataset: `str`
    The dataset CSV/TSV path in the distributed file system.
    It expects a test dataset.

    filepath_sub_dataset: `str`
    The dataset CSV/TSV path in the distributed file system.
    It expects a prediction dataset.

    Returns
    -------
    `str` The name for the plot image in the DFS.
    """
    df = pd.read_csv(filepath_sub_dataset)
    plt.figure(figsize=(10, 6))
    plt.scatter(df.index, df['negative_conf'], alpha=0.5, label='Negative')
    plt.scatter(df.index, df['neutral_conf'], alpha=0.5, label='Neutral')
    plt.scatter(df.index, df['positive_conf'], alpha=0.5, label='Positive')
    plt.title("Confidence Scores Across All Samples")
    plt.xlabel("Sample Index")
    plt.ylabel("Confidence Score")
    plt.legend()
    plt.tight_layout()

    # Save the plot to a PNG file
    filename = f"/result/confidence_scores.png"
    plt.savefig(filename)

    return "confidence_scores.png"


def generate_heatmap(filepath_test_dataset: str, filepath_sub_dataset: str) -> str:
    """
    Generate a confusion matrix heatmap based on true and predicted sentiment labels.
    
    Parameters
    ----------
    filepath_test_dataset : str
        The CSV file path for the test dataset that contains the true labels in the 'target' column.
    
    filepath_sub_dataset : str
        The CSV file path for the prediction dataset that contains the predicted labels in the 'target' column.
    
    Returns
    -------
    str
        The file path for the saved confusion matrix plot image in the DFS.
    """
    # Load the datasets
    df_true = pd.read_csv(filepath_test_dataset)
    df_pred = pd.read_csv(filepath_sub_dataset)
    
    # Extract true and predicted labels from the 'target' column
    y_true = df_true['target'].values
    y_pred = df_pred['target'].values
    
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Determine the label order (e.g., [0, 2, 4] for negative, neutral, positive)
    labels = sorted(set(y_true) | set(y_pred))
    
    # Create the heatmap plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    
    # Save the plot to a PNG file in the DFS
    filename = "/result/confusion_matrix.png"
    plt.savefig(filename)
    plt.close()
    
    return filename