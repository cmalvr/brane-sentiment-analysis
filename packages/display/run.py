#!/usr/bin/python3
'''
Entrypoint for the visualization package.
'''
import os
import sys
import yaml

import json

from display import generate_prediction_plot


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

    if command == "generate_prediction_plot":
        filepath_sub_dataset = f"{json.loads(os.environ['FILEPATH_SUBMISSION'])}/submission.csv"
        output = generate_prediction_plot(filepath_sub_dataset)
        return
    
if __name__ == '__main__':
    main()