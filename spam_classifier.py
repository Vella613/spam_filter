
import re
import string
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import tkinter as tk
from tkinter import messagebox

# Ensure nltk resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Set random seed for reproducibility
random_state = 42

# Load the dataset into a pandas DataFrame
df = pd.read_csv('reduced_dataset_20030228_from_unzipped.csv')

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Define a function for preprocessing email text with optional steps
def preprocess_email(text, remove_punctuation=True, lowercase=True, remove_urls=True, remove_numbers=True, remove_headers=True):
    if remove_headers:
        text = re.sub(r'^(From:|To:|Subject:|Date:)[^\n]*\n', '', text, flags=re.MULTILINE)
    if remove_numbers:
        text = re.sub(r'\d+', 'NUMBER', text)
    if remove_urls:
        text = re.sub(r'http\S+|www\S+', 'URL', text)
    if lowercase:
        text = text.lower()
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Apply the preprocessing function to the 'subject' column
df['cleaned_text'] = df['subject'].apply(preprocess_email)

# Encode the target labels ('class') as numerical values
le = LabelEncoder()
df['label'] = le.fit_transform(df['class'])

# Split the data into training and testing sets
X = df['cleaned_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# Define the pipelines for each model
pipeline_lr = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words='english', binary=True, max_features=10000, ngram_range=(1, 3))),
    ('oversample', RandomOverSampler(random_state=random_state)),
    ('classifier', LogisticRegression(solver='liblinear', class_weight='balanced'))
])

pipeline_rf = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words='english', binary=True, max_features=10000, ngram_range=(1, 3))),
    ('oversample', RandomOverSampler(random_state=random_state)),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=random_state))
])

pipeline_svc = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words='english', binary=True, max_features=10000, ngram_range=(1, 3))),
    ('oversample', RandomOverSampler(random_state=random_state)),
    ('classifier', SVC(class_weight='balanced', probability=True, random_state=random_state))
])

# Define a common parameter grid that applies to all models
param_grid = {
    'vectorizer__max_df': [0.75, 0.85, 0.95], 
    'vectorizer__min_df': [1, 2], 
    'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)]
}

# Define specific parameter grids for each classifier
param_grid_lr = {**param_grid, 'classifier__C': [0.01, 0.1, 1, 10, 100]}
param_grid_rf = {**param_grid, 'classifier__n_estimators': [100, 200, 500], 'classifier__max_depth': [10, 20, 50]}
param_grid_svc = {**param_grid, 'classifier__C': [0.01, 0.1, 1, 10, 100]}

# Perform GridSearchCV for hyperparameter tuning for each model
grid_search_lr = GridSearchCV(pipeline_lr, param_grid_lr, cv=3, verbose=1, n_jobs=-1)
grid_search_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=3, verbose=1, n_jobs=-1)
grid_search_svc = GridSearchCV(pipeline_svc, param_grid_svc, cv=3, verbose=1, n_jobs=-1)

# Fit the models
grid_search_lr.fit(X_train, y_train)
grid_search_rf.fit(X_train, y_train)
grid_search_svc.fit(X_train, y_train)

# Define a function to evaluate the model and calculate classification metrics
def evaluate_model(grid_search, X_test, y_test):
    y_pred = grid_search.best_estimator_.predict(X_test)  
    accuracy = accuracy_score(y_test, y_pred)  
    precision = precision_score(y_test, y_pred)  
    recall = recall_score(y_test, y_pred)  
    f1 = f1_score(y_test, y_pred)  
    return accuracy, precision, recall, f1, y_pred

# Store the results for comparison
results = {}

# Evaluate Logistic Regression
results['Logistic Regression'] = evaluate_model(grid_search_lr, X_test, y_test)

# Evaluate Random Forest
results['Random Forest'] = evaluate_model(grid_search_rf, X_test, y_test)

# Evaluate SVC
results['SVC'] = evaluate_model(grid_search_svc, X_test, y_test)

# Print classification reports for all models
for model_name, result in results.items():
    print(f"\nClassification Report for {model_name}:")
    y_pred = result[4]  
    print(classification_report(y_test, y_pred, target_names=le.classes_))

# Plot confusion matrices for all models
plt.figure(figsize=(15, 5))

for i, (model_name, result) in enumerate(results.items(), 1):
    y_pred = result[4]
    cm = confusion_matrix(y_test, y_pred)
    
    plt.subplot(1, 3, i)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

plt.tight_layout()
plt.show()

# Plot ROC curve for all models
plt.figure(figsize=(15, 5))

for i, (model_name, result) in enumerate(results.items(), 1):
    if model_name == 'Logistic Regression':
        y_prob = grid_search_lr.best_estimator_.predict_proba(X_test)[:, 1]
    elif model_name == 'Random Forest':
        y_prob = grid_search_rf.best_estimator_.predict_proba(X_test)[:, 1]
    elif model_name == 'SVC':
        y_prob = grid_search_svc.best_estimator_.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)  

    plt.subplot(1, 3, i)
    plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title(f'ROC Curve - {model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')

plt.tight_layout()
plt.show()

# Function to classify a message
def classify_message():
    # Get the message from the input field
    message = message_entry.get()

    # Preprocess the message using the same preprocessing function as before
    cleaned_message = preprocess_email(message)

    # Use the best model to make predictions
    prediction = grid_search_lr.best_estimator_.predict([cleaned_message])  # Using Logistic Regression model here
    predicted_label = le.inverse_transform(prediction)[0]  # Convert numeric prediction back to text

    # Display the result in a messagebox
    messagebox.showinfo("Classification Result", f"The message is classified as: {predicted_label}")

# Initialize the main window for the GUI
root = tk.Tk()
root.title("Spam Classifier")

# Add a label
label = tk.Label(root, text="Enter your message to classify as Spam or Ham:")
label.pack(pady=10)

# Add a text entry field
message_entry = tk.Entry(root, width=50)
message_entry.pack(pady=10)

# Add a classify button
classify_button = tk.Button(root, text="Classify", command=classify_message)
classify_button.pack(pady=20)

# Start the GUI event loop
root.mainloop()

