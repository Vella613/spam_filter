import os
import re
import pandas as pd

def extract_subject_date_and_delivered_from_email(email_content):
    """
    Extracts the 'Subject', 'Date', and 'Delivered-To' fields from email content using regular expressions.

    Args:
        email_content (str): The raw content of an email.

    Returns:
        tuple: A tuple containing the extracted subject, date, and delivered-to fields. 
               Returns None for a field if it cannot be found.
    """
    subject_re = r"Subject:\s*(.*)"
    date_re = r"Date:\s*(.*)"
    delivered_to_re = r"Delivered-To:\s*(.*)"
    
    # Performs regex search for each field

    subject_match = re.search(subject_re, email_content, re.MULTILINE | re.IGNORECASE)
    date_match = re.search(date_re, email_content, re.MULTILINE | re.IGNORECASE)
    delivered_to_match = re.search(delivered_to_re, email_content, re.MULTILINE | re.IGNORECASE)

    # Extracts and strips leading/trailing whitespace if matches are found

    subject = subject_match.group(1).strip() if subject_match else None
    date = date_match.group(1).strip() if date_match else None
    delivered_to = delivered_to_match.group(1).strip() if delivered_to_match else None

    return subject, date, delivered_to

def load_emails_from_folder(folder_path, label):
    """
    Loads email files from a specified folder, extract fields, and classify them by label.

    Args:
        folder_path (str): Path to the folder containing email files.
        label (str): Classification label for the emails (e.g., 'ham' or 'spam').

    Returns:
        list: A list of tuples containing subject, date, delivered-to, and label for each email.
    """
    emails = []
    print(f"Loading emails from {label} folder: {folder_path}")
    
    # Traverses through the directory tree to locate email files

    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            
            try:
                # Reads email content from file

                with open(file_path, 'r', errors='ignore') as f:
                    email_content = f.read()

                    # Extracts required fields from email content
                    subject, date, delivered_to = extract_subject_date_and_delivered_from_email(email_content)

                    # Only includes emails with all required fields
                    if subject and date and delivered_to:
                        emails.append((subject, date, delivered_to, label))
                    else:
                        print(f"Skipped {file_path}: Missing required fields.")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    print(f"Extracted {len(emails)} emails from {label} folder.")
    return emails

def load_all_emails_from_folder(main_folder):
    """
    Loads emails from 'ham' and 'spam' folders, extract fields, and compile them into a DataFrame.

    Args:
        main_folder (str): Path to the main dataset folder containing 'ham' and 'spam' subfolders.

    Returns:
        pd.DataFrame: A DataFrame with columns for subject, date, delivered-to, and class.
                      Returns None if required folders are missing.
    """
    # Defines paths for ham and spam folders

    ham_folder = os.path.join(main_folder, 'unzipped', 'easy_ham')
    spam_folder = os.path.join(main_folder, 'unzipped', 'spam')

    # Checks if both folders exist

    if not os.path.exists(ham_folder) or not os.path.exists(spam_folder):
        print(f"Error: One or both folders ({ham_folder}, {spam_folder}) are missing.")
        return None

    # Loads and limits to the first 500 emails from each folder

    ham_emails = load_emails_from_folder(ham_folder, 'ham')[:500]
    spam_emails = load_emails_from_folder(spam_folder, 'spam')[:500]


    # Combines ham and spam emails into a single list

    emails = ham_emails + spam_emails
    
    # Creates a DataFrame from the combined data

    df = pd.DataFrame(emails, columns=["subject", "date", "delivered_to", "class"])

    return df

if __name__ == "__main__":
   # Determines the script's directory and defines the dataset folder path

    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_folder = os.path.join(script_dir, 'dataset')

   # Loads emails into a DataFrame

    df = load_all_emails_from_folder(main_folder)

    if df is not None:
        # Checks if all subjects are empty after extraction

        if df["subject"].str.strip().eq("").all():
            print("Error: All subjects are empty after extraction.")
            exit(1)
        
        # Display a preview of the DataFrame
        print("DataFrame preview:")
        print(df.head())

        # Saves the DataFrame to a CSV file
        output_file = "dataset_20030228_from_unzipped.csv"
        df.to_csv(output_file, index=False)
        print(f"Data saved to '{output_file}'.")
