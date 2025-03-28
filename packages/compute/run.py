#!/usr/bin/python3
'''
Entrypoint for the sentiment analysis compute package.
'''
import os
import sys
import json
import yaml

from model import create_submission, train_model
from preprocess import clean, remove_stopwords, tokenize


def run_dataset_action(cmd: str, filepath: str):
    """
    Runs generic dataset preprocessing action.

    Parameters
    ----------
    cmd: `str`
    The action name.

    filepath: `str`
    The dataset filepath in the DFS.
    """
    return {
        "clean": clean,
        "tokenize": tokenize,
        "remove_stopwords": remove_stopwords,
    }[cmd](filepath)


def print_output(data: dict):
    """
    Creates a marked section in the standard output
    of the container in order for Brane to isolate the result.

    Parameters
    ----------
    data: `dict`
    Any valid Python dictionary that is YAML serializable.
    """
    print("--> START CAPTURE")
    print(yaml.dump(data))
    print("--> END CAPTURE")


def main():
    command = sys.argv[1]

    if command == "train_model":
        # Get paths from environment variables
        filepath_dataset = f"{json.loads(os.environ['FILEPATH_DATASET'])}/dataset.csv"
        filepath_model = train_model(filepath_dataset)
        return

    if command == "create_submission":
        # Get paths from environment variables
        filepath_dataset = f"{json.loads(os.environ['FILEPATH_DATASET'])}/dataset.csv"
        filepath_model = f"{json.loads(os.environ['FILEPATH_MODEL'])}/model.pickle"
        filepath_submission = create_submission(filepath_dataset, filepath_model)
        return

    # For preprocessing actions (clean, tokenize, remove_stopwords)
    filepath_in = json.loads(os.environ["FILEPATH"])
    filepath_out = run_dataset_action(command, filepath_in)


if __name__ == '__main__':
    main()