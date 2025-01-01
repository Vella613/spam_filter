import tarfile
import os

def extract_tar_file(tar_file_path, extraction_dir):
    """
    Extracts the given tar.bz2 file to the specified directory.
    :param tar_file_path: Path to the tar.bz2 file
    :param extraction_dir: Directory where the file should be extracted
    """
    # Ensures the extraction directory exists
    os.makedirs(extraction_dir, exist_ok=True)

    # Checks if the tar.bz2 file exists
    if not os.path.exists(tar_file_path):
        print(f"The file {tar_file_path} does not exist.")
        return

    # Opens and extracts the tar.bz2 file
    with tarfile.open(tar_file_path, 'r:bz2') as tar_ref:
        tar_ref.extractall(extraction_dir)

    print(f"Dataset extracted to {extraction_dir}")


# File paths
tar_file_path_ham = r'dataset\20030228_easy_ham.tar.bz2'
tar_file_path_spam = r'dataset\20030228_spam.tar.bz2'

# Extractions directories
extraction_dir_ham = r'dataset\unzipped'
extraction_dir_spam = r'dataset\unzipped'

# Extracts the datasets
extract_tar_file(tar_file_path_ham, extraction_dir_ham)
extract_tar_file(tar_file_path_spam, extraction_dir_spam)
