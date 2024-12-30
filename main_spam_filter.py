import pandas as pd  # For data manipulation and analysis
import re  # For regular expressions to process text
import string  # To handle and manipulate strings
from sklearn.model_selection import train_test_split, RandomizedSearchCV  # For splitting data and hyperparameter tuning
from sklearn.feature_extraction.text import TfidfVectorizer  # For text feature extraction
from sklearn.linear_model import LogisticRegression  # For logistic regression classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix  # For evaluation metrics
from sklearn.preprocessing import LabelEncoder  # For encoding target labels
from imblearn.over_sampling import RandomOverSampler  # For handling class imbalance
from imblearn.pipeline import Pipeline  # To create pipelines for data processing and modeling
import seaborn as sns  # For creating visualizations
import matplotlib.pyplot as plt  # For plotting graphs
import tkinter as tk  # For creating the GUI
from tkinter import ttk  # For advanced GUI widgets

# Load the dataset into a pandas DataFrame
df = pd.read_csv('reduced_dataset_20030228_from_unzipped.csv')

# Define a function for preprocessing email text
def preprocess_email(text):
    """
    Cleans and preprocesses email text by:
    - Removing headers between "Subject:" and "Date:"
    - Replacing numbers with 'NUMBER'
    - Replacing URLs with 'URL'
    - Converting text to lowercase
    - Removing punctuation
    """
    text = re.sub(r'(?<=Subject:)(.*?)(?=Date:)', '', text, flags=re.DOTALL)  # Remove subject headers
    text = re.sub(r'\d+', 'NUMBER', text)  # Replace numbers with 'NUMBER'
    text = re.sub(r'http\S+|www\S+', 'URL', text)  # Replace URLs with 'URL'
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text

# Apply the preprocessing function to the 'subject' column of the dataset
df['cleaned_text'] = df['subject'].apply(preprocess_email)

# Encode the target labels ('class') as numerical values using LabelEncoder
le = LabelEncoder()
df['label'] = le.fit_transform(df['class'])  # Spam = 1, Ham = 0

# Split the data into training and testing sets (80% training, 20% testing)
X = df['cleaned_text']  # Features: cleaned email text
y = df['label']  # Target labels: spam/ham
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a pipeline for logistic regression with hyperparameter tuning
pipeline_lr = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words='english')),  # Convert text to TF-IDF features
    ('oversample', RandomOverSampler(random_state=42)),  # Handle class imbalance
    ('classifier', LogisticRegression(solver='liblinear'))  # Logistic regression model
])

# Define a parameter grid for hyperparameter tuning
param_grid_lr = {
    'vectorizer__max_df': [0.75, 0.85, 0.95],  # Maximum document frequency
    'vectorizer__min_df': [1, 2],  # Minimum document frequency
    'vectorizer__ngram_range': [(1, 1), (1, 2)],  # N-grams: unigrams or bigrams
    'classifier__C': [0.1, 1, 10]  # Regularization strength
}

# Perform random search for hyperparameter tuning
random_search_lr = RandomizedSearchCV(
    pipeline_lr, param_distributions=param_grid_lr, n_iter=10, cv=3, verbose=1, n_jobs=-1, random_state=42
)
random_search_lr.fit(X_train, y_train)

# Define a function to predict spam/ham messages using the trained model
def predict_message():
    """
    Takes user input from the GUI, preprocesses it, and predicts whether the message is spam or ham.
    """
    user_input = entry_message.get()
    if user_input.strip():  # Check if input is not empty
        processed_message = preprocess_email(user_input)  # Preprocess the input message
        prediction = random_search_lr.best_estimator_.predict([processed_message])[0]  # Predict class
        result = "Spam" if prediction == 1 else "Ham"  # Map prediction to class label
        label_result.config(text=f"Prediction: {result}")  # Display result in the GUI
    else:
        label_result.config(text="Please enter a valid message.")  # Handle empty input

# Create the GUI for spam detection
root = tk.Tk()
root.title("Spam Detector")

# Create a frame for GUI components
frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Input field for message
label_message = ttk.Label(frame, text="Enter your message:")
label_message.grid(row=0, column=0, pady=5, sticky=tk.W)
entry_message = ttk.Entry(frame, width=50)
entry_message.grid(row=1, column=0, pady=5, sticky=tk.W)

# Button to trigger prediction
button_predict = ttk.Button(frame, text="Predict", command=predict_message)
button_predict.grid(row=2, column=0, pady=10, sticky=tk.W)

# Label to display prediction results
label_result = ttk.Label(frame, text="")
label_result.grid(row=3, column=0, pady=5, sticky=tk.W)

# Run the GUI loop
root.mainloop()

# Evaluate the model on the test set
print("\nEvaluating Logistic Regression...")
y_pred_best = random_search_lr.best_estimator_.predict(X_test)  # Predict on test set
accuracy = accuracy_score(y_test, y_pred_best)  # Calculate accuracy
precision = precision_score(y_test, y_pred_best)  # Calculate precision
recall = recall_score(y_test, y_pred_best)  # Calculate recall
f1 = f1_score(y_test, y_pred_best)  # Calculate F1 score

# Print evaluation metrics
print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
print("\nClassification results:")
print(classification_report(y_test, y_pred_best, target_names=le.classes_))  # Display classification report

# Create and display a confusion matrix
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
