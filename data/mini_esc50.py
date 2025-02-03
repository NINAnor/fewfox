#!/usr/bin/env python3

import pandas as pd
import shutil
import hydra
from pathlib import Path

def few_shot_sample(df: pd.DataFrame, classes: list, n_samples: int, seed: int) -> pd.DataFrame:
    return pd.concat(
        [df[df["category"] == c].sample(n_samples, random_state=seed) for c in classes]
    )

def copy_to_folder(df: pd.DataFrame, target_folder: Path):
    for row in df.itertuples(index=False):
        shutil.copy(row.filepath, target_folder / row.filename)

def split_data(df: pd.DataFrame, train_samples: int, val_samples: int):
    train_list, val_list, test_list = [], [], []
    for _, group in df.groupby("category"):
        train = group.sample(n=train_samples)
        remaining = group.drop(train.index)
        val = remaining.sample(n=val_samples)
        test = remaining.drop(val.index)
        train_list.append(train)
        val_list.append(val)
        test_list.append(test)
    return pd.concat(train_list), pd.concat(val_list), pd.concat(test_list)

def delete_path(path: Path):
    if path.exists():
        shutil.rmtree(path)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    # Create Path objects from configuration
    root_path = Path(cfg.paths.ROOT_PATH)
    csv_file = Path(cfg.paths.CSV_FILE_ESC50)
    target_path = Path(cfg.paths.MINI_ESC50_PATH)

    # Read the CSV and create a new column with full file paths
    df = pd.read_csv(csv_file)
    df["filepath"] = df["filename"].apply(lambda fname: root_path / "ESC-50-master" / "audio" / fname)

    # Define the classes to sample
    classes = ["dog", "cat", "chirping_birds", "crying_baby", "crow"]

    # Create the few-shot dataset (40 samples per class) and split it into train, validation, and test sets
    few_shot_df = few_shot_sample(df, classes, n_samples=40, seed=42)
    train_df, val_df, test_df = split_data(few_shot_df, train_samples=5, val_samples=5)

    # Delete the target folder if it exists and create required subdirectories
    delete_path(target_path)
    for subfolder in ["audio/train", "audio/val", "audio/test", "meta"]:
        (target_path / subfolder).mkdir(parents=True, exist_ok=True)

    # Save the DataFrames as CSV files
    train_df.to_csv(target_path / "meta" / "esc50mini_train.csv", index=False)
    val_df.to_csv(target_path / "meta" / "esc50mini_val.csv", index=False)
    test_df.to_csv(target_path / "meta" / "esc50mini_test.csv", index=False)

    # Copy the audio files into the respective folders
    copy_to_folder(train_df, target_path / "audio/train")
    copy_to_folder(val_df, target_path / "audio/val")
    copy_to_folder(test_df, target_path / "audio/test")

if __name__ == "__main__":
    main()
