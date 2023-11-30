import os
import random
from sklearn.model_selection import train_test_split
import shutil

def train_test_split_and_move(classification_folder1, classification_folder2, test_size=0.2, random_state=None):
    """
    Perform train-test split for two classification folders containing JPG files
    and move the files into train and test folders.

    Parameters:
    - classification_folder1: Path to the first classification folder.
    - classification_folder2: Path to the second classification folder.
    - test_size: Proportion of files to include in the test split (default is 0.2).
    - random_state: Seed for random number generator (optional).

    Returns:
    - None
    """

    # Create train and test folders for each classification
    train_folder1 = os.path.join(classification_folder1, "train")
    test_folder1 = os.path.join(classification_folder1, "test")
    train_folder2 = os.path.join(classification_folder2, "train")
    test_folder2 = os.path.join(classification_folder2, "test")

    os.makedirs(train_folder1, exist_ok=True)
    os.makedirs(test_folder1, exist_ok=True)
    os.makedirs(train_folder2, exist_ok=True)
    os.makedirs(test_folder2, exist_ok=True)

    # List all JPG files in the classification folders
    files1 = [f for f in os.listdir(classification_folder1) if f.endswith(".png")]
    files2 = [f for f in os.listdir(classification_folder2) if f.endswith(".png")]

    # Perform train-test split for each classification
    train_files1, test_files1 = train_test_split(files1, test_size=test_size, random_state=random_state)
    train_files2, test_files2 = train_test_split(files2, test_size=test_size, random_state=random_state)

    # Move files to the appropriate train and test folders
    for file in train_files1:
        src = os.path.join(classification_folder1, file)
        dst = os.path.join(train_folder1, file)
        shutil.move(src, dst)

    for file in test_files1:
        src = os.path.join(classification_folder1, file)
        dst = os.path.join(test_folder1, file)
        shutil.move(src, dst)

    for file in train_files2:
        src = os.path.join(classification_folder2, file)
        dst = os.path.join(train_folder2, file)
        shutil.move(src, dst)

    for file in test_files2:
        src = os.path.join(classification_folder2, file)
        dst = os.path.join(test_folder2, file)
        shutil.move(src, dst)

# Example usage:
train_test_split_and_move("../../Driver Drowsiness Dataset (DDD)/Drowsy", "../../Driver Drowsiness Dataset (DDD)/Non Drowsy", test_size=0.2, random_state=42)
