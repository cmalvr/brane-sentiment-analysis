import matplotlib.pyplot as plt
import pandas as pd

def generate_prediction_plot(filepath_sub_dataset: str) -> str:
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
