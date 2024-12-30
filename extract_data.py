import os
import re
import pandas as pd

def extract_subject_date_and_delivered_from_email(email_content):
    """
    Extract subject, date, and delivered-to from email content using regex.
    """
    subject_re = r"Subject:\s*(.*)"
    date_re = r"Date:\s*(.*)"
    delivered_to_re = r"Delivered-To:\s*(.*)"

    subject_match = re.search(subject_re, email_content, re.MULTILINE | re.IGNORECASE)
    date_match = re.search(date_re, email_content, re.MULTILINE | re.IGNORECASE)
    delivered_to_match = re.search(delivered_to_re, email_content, re.MULTILINE | re.IGNORECASE)

    subject = subject_match.group(1).strip() if subject_match else None
    date = date_match.group(1).strip() if date_match else None
    delivered_to = delivered_to_match.group(1).strip() if delivered_to_match else None

    return subject, date, delivered_to

def load_emails_from_folder(folder_path, label):
    """
    Load emails from the specified folder and extract fields using regex.
    """
    emails = []
    print(f"Loading emails from {label} folder: {folder_path}")

    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            
            try:
                with open(file_path, 'r', errors='ignore') as f:
                    email_content = f.read()

                    subject, date, delivered_to = extract_subject_date_and_delivered_from_email(email_content)

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
    Load and combine emails from ham and spam folders into a DataFrame.
    """
    ham_folder = os.path.join(main_folder, 'unzipped', 'easy_ham')
    spam_folder = os.path.join(main_folder, 'unzipped', 'spam')

    if not os.path.exists(ham_folder) or not os.path.exists(spam_folder):
        print(f"Error: One or both folders ({ham_folder}, {spam_folder}) are missing.")
        return None

    ham_emails = load_emails_from_folder(ham_folder, 'ham')[:500]
    spam_emails = load_emails_from_folder(spam_folder, 'spam')[:500]

    emails = ham_emails + spam_emails
    df = pd.DataFrame(emails, columns=["subject", "date", "delivered_to", "class"])

    return df

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_folder = os.path.join(script_dir, 'dataset')

    df = load_all_emails_from_folder(main_folder)

    if df is not None:
        if df["subject"].str.strip().eq("").all():
            print("Error: All subjects are empty after extraction.")
            exit(1)

        print("DataFrame preview:")
        print(df.head())

        output_file = "dataset_20030228_from_unzipped.csv"
        df.to_csv(output_file, index=False)
        print(f"Data saved to '{output_file}'.")
