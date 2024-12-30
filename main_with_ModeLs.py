import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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
import tkinter as tk  # For creating the GUI
from tkinter import ttk  # For advanced GUI widgets

# Set random seed for reproducibility
random_state = 42

# Load the dataset into a pandas DataFrame
df = pd.read_csv('reduced_dataset_20030228_from_unzipped.csv')

# Define a function for preprocessing email text
def preprocess_email(text):
    text = re.sub(r'(?<=Subject:)(.*?)(?=Date:)', '', text, flags=re.DOTALL)
    text = re.sub(r'\d+', 'NUMBER', text)
    text = re.sub(r'http\S+|www\S+', 'URL', text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Apply the preprocessing function to the 'subject' column
df['cleaned_text'] = df['subject'].apply(preprocess_email)

# Encode the target labels ('class') as numerical values
le = LabelEncoder()
df['label'] = le.fit_transform(df['class'])  # Spam = 1, Ham = 0

# Split the data into training and testing sets
X = df['cleaned_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# Define the pipelines for each model
pipeline_lr = Pipeline([('vectorizer', TfidfVectorizer(stop_words='english')), 
                        ('oversample', RandomOverSampler(random_state=random_state)), 
                        ('classifier', LogisticRegression(solver='liblinear'))])

pipeline_rf = Pipeline([('vectorizer', TfidfVectorizer(stop_words='english')), 
                        ('oversample', RandomOverSampler(random_state=random_state)), 
                        ('classifier', RandomForestClassifier(random_state=random_state))])

pipeline_svc = Pipeline([('vectorizer', TfidfVectorizer(stop_words='english')), 
                         ('oversample', RandomOverSampler(random_state=random_state)), 
                         ('classifier', SVC(probability=True, random_state=random_state))])

# Perform random search for hyperparameter tuning
param_grid_lr = {'vectorizer__max_df': [0.75, 0.85, 0.95], 'vectorizer__min_df': [1, 2], 'vectorizer__ngram_range': [(1, 1), (1, 2)], 'classifier__C': [0.1, 1, 10]}
param_grid_rf = {'vectorizer__max_df': [0.75, 0.85, 0.95], 'vectorizer__min_df': [1, 2], 'vectorizer__ngram_range': [(1, 1), (1, 2)], 'classifier__n_estimators': [50, 100, 200], 'classifier__max_depth': [10, 20, None]}
param_grid_svc = {'vectorizer__max_df': [0.75, 0.85, 0.95], 'vectorizer__min_df': [1, 2], 'vectorizer__ngram_range': [(1, 1), (1, 2)], 'classifier__C': [0.1, 1, 10], 'classifier__kernel': ['linear', 'rbf']}

# Random seaArch tuning
random_search_lr = RandomizedSearchCV(pipeline_lr, param_distributions=param_grid_lr, n_iter=10, cv=3, verbose=1, n_jobs=-1, random_state=random_state)
random_search_rf = RandomizedSearchCV(pipeline_rf, param_distributions=param_grid_rf, n_iter=10, cv=3, verbose=1, n_jobs=-1, random_state=random_state)
random_search_svc = RandomizedSearchCV(pipeline_svc, param_distributions=param_grid_svc, n_iter=10, cv=3, verbose=1, n_jobs=-1, random_state=random_state)

# Fit the models
random_search_lr.fit(X_train, y_train)
random_search_rf.fit(X_train, y_train)
random_search_svc.fit(X_train, y_train)

# Evaluate function that stores results
def evaluate_model(random_search, X_test, y_test):
    y_pred = random_search.best_estimator_.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1, y_pred

# Store results for comparison
results = {}
y_pred_lr = None
results['Logistic Regression'] = evaluate_model(random_search_lr, X_test, y_test)
y_pred_lr = results['Logistic Regression'][4]  # Saving the predictions for LR

results['Random Forest'] = evaluate_model(random_search_rf, X_test, y_test)
results['Support Vector Classifier'] = evaluate_model(random_search_svc, X_test, y_test)

# Best model for confusion matrix and ROC curve (Logistic Regression)
best_model_name = 'Logistic Regression'
print(f"Best Model: {best_model_name}")
best_model_results = results[best_model_name]
print(f"Accuracy: {best_model_results[0]:.4f}, Precision: {best_model_results[1]:.4f}, Recall: {best_model_results[2]:.4f}, F1 Score: {best_model_results[3]:.4f}")

# Classification report for Logistic Regression
print("\nClassification Report for Logistic Regression:")
y_pred_lr = random_search_lr.best_estimator_.predict(X_test)  # Predictions for LR
print(classification_report(y_test, y_pred_lr, target_names=le.classes_))

# Display confusion matrix for Logistic Regression
cm = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plot ROC curve for Logistic Regression
y_prob_lr = random_search_lr.best_estimator_.predict_proba(X_test)[:, 1]  # Probabilities for the positive class (spam)
fpr, tpr, thresholds = roc_curve(y_test, y_prob_lr)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Logistic Regression')
plt.legend(loc='lower right')
plt.show()

# Display classification report for other models
print("\nClassification Report for Random Forest:")
y_pred_rf = random_search_rf.best_estimator_.predict(X_test)  # Predictions for Random Forest
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

print("\nClassification Report for Support Vector Classifier:")
y_pred_svc = random_search_svc.best_estimator_.predict(X_test)  # Predictions for SVC
print(classification_report(y_test, y_pred_svc, target_names=le.classes_))

# Display comparison for other models
print("\nComparison of all models:")
for model_name, metrics in results.items():
    if model_name != 'Logistic Regression':
        print(f"{model_name}: Accuracy = {metrics[0]:.4f}, Precision = {metrics[1]:.4f}, Recall = {metrics[2]:.4f}, F1 = {metrics[3]:.4f}")



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
