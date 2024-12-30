import os
import re
import pandas as pd

# Function to balance the dataset by limiting to 500 ham and 500 spam entries
def balance_dataset(input_csv, output_csv):
    # Read the already extracted data from the CSV file
    df = pd.read_csv(input_csv)

    # Check if the CSV was read correctly
    print("Data preview from CSV:")
    print(df.head())  # Print the first few rows for verification

    # Separate ham and spam emails
    ham_df = df[df['class'] == 'ham']
    spam_df = df[df['class'] == 'spam']

    # Check the size of ham and spam data
    print(f"Total ham emails: {len(ham_df)}")
    print(f"Total spam emails: {len(spam_df)}")

    # Limit both ham and spam to 500 emails each
    ham_df = ham_df.head(400)
    spam_df = spam_df.head(400)

    # Combine the ham and spam dataframes
    balanced_df = pd.concat([ham_df, spam_df])

    # Shuffle the combined dataframe to mix ham and spam emails
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save the balanced dataset to a new CSV file
    balanced_df.to_csv(output_csv, index=False)
    print(f"Balanced dataset saved to {output_csv}")
if __name__ == "__main__":
    input_csv = "dataset_20030228_from_unzipped.csv"  # Path to the extracted CSV file
    output_csv = "reduced_dataset_20030228_from_unzipped.csv"  # Path to save the balanced dataset

    # Balance the dataset and save it
    balance_dataset(input_csv, output_csv)