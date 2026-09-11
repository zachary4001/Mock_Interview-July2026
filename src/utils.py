from pathlib import Path


def create_output_folders(*folders):

    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)