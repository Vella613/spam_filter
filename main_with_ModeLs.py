import pandas as pd  # For data manipulation and analysis
import re  # For regular expressions to clean text
import string  # For text processing (removing punctuation)
from sklearn.model_selection import train_test_split, RandomizedSearchCV  # For data splitting and hyperparameter tuning
from sklearn.feature_extraction.text import TfidfVectorizer  # For text vectorization
from sklearn.linear_model import LogisticRegression  # Logistic Regression classifier
from sklearn.ensemble import RandomForestClassifier  # Random Forest classifier
from sklearn.svm import SVC  # Support Vector Machine classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc  # Metrics
from imblearn.over_sampling import RandomOverSampler  # For handling imbalanced datasets
from imblearn.pipeline import Pipeline  # for creating machine learning pipelines
import seaborn as sns  # For visualization
import matplotlib.pyplot as plt  # For plotting
from sklearn.preprocessing import LabelEncoder  # For encoding labels into numerical values
import tkinter as tk  # For GUI creation
from tkinter import ttk  # For advanced GUI widgets

# Setting random seed for reproducibility
random_state = 42

# Loading the dataset into a pandas DataFrame
df = pd.read_csv('reduced_dataset_20030228_from_unzipped.csv')  # Replacing with the actual dataset path

# Function to preprocess email text
def preprocess_email(text):
    """
    Preprocesses an email by removing unnecessary elements, replacing numbers and URLs, converting to lowercase,
    and removing punctuation.
    """
    text = re.sub(r'(?<=Subject:)(.*?)(?=Date:)', '', text, flags=re.DOTALL)  # Removing text between 'Subject' and 'Date'
    text = re.sub(r'\d+', 'NUMBER', text)  # Replacing digits with 'NUMBER'
    text = re.sub(r'http\S+|www\S+', 'URL', text)  # Replacing URLs with 'URL'
    text = text.lower()  # Converting text to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Removing punctuation
    return text

# Applying preprocessing to the 'subject' column
df['cleaned_text'] = df['subject'].apply(preprocess_email)  # Cleaning the email subjects

# Encoding target labels ('class') into numerical values
le = LabelEncoder()
df['label'] = le.fit_transform(df['class'])  # Encoding 'spam' as 1 and 'ham' as 0

# Splitting dataset into training and testing sets
X = df['cleaned_text']  # Features (processed email text)
y = df['label']  # Labels (spam or ham)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)  # 80-20 split

# Defining pipelines for each model with oversampling and TF-IDF vectorization
pipeline_lr = Pipeline([('vectorizer', TfidfVectorizer(stop_words='english')),  # Converting text into numerical features
                        ('oversample', RandomOverSampler(random_state=random_state)),  # Handling class imbalance
                        ('classifier', LogisticRegression(solver='liblinear'))])  # Logistic Regression model

pipeline_rf = Pipeline([('vectorizer', TfidfVectorizer(stop_words='english')),
                        ('oversample', RandomOverSampler(random_state=random_state)),
                        ('classifier', RandomForestClassifier(random_state=random_state))])  # Random Forest model

pipeline_svc = Pipeline([('vectorizer', TfidfVectorizer(stop_words='english')),
                         ('oversample', RandomOverSampler(random_state=random_state)),
                         ('classifier', SVC(probability=True, random_state=random_state))])  # SVM model

# Defining hyperparameter grids for RandomizedSearchCV tuning
param_grid_lr = {
    'vectorizer__max_df': [0.75, 0.85, 0.95],  # Max document frequency
    'vectorizer__min_df': [1, 2],  # Min document frequency
    'vectorizer__ngram_range': [(1, 1), (1, 2)],  # Unigrams or bigrams
    'classifier__C': [0.1, 1, 10]  # Regularization strength
}

param_grid_rf = {
    'vectorizer__max_df': [0.75, 0.85, 0.95],
    'vectorizer__min_df': [1, 2],
    'vectorizer__ngram_range': [(1, 1), (1, 2)],
    'classifier__n_estimators': [50, 100, 200],  # Number of trees in the forest
    'classifier__max_depth': [10, 20, None]  # Maximum depth of trees
}

param_grid_svc = {
    'vectorizer__max_df': [0.75, 0.85, 0.95],
    'vectorizer__min_df': [1, 2],
    'vectorizer__ngram_range': [(1, 1), (1, 2)],
    'classifier__C': [0.1, 1, 10],  # Regularization strength
    'classifier__kernel': ['linear', 'rbf']  # Kernel types for SVM
}

# Performing RandomizedSearchCV for each model
random_search_lr = RandomizedSearchCV(pipeline_lr, param_distributions=param_grid_lr, n_iter=10, cv=3, verbose=1, n_jobs=-1, random_state=random_state)  # Tune Logistic Regression
random_search_rf = RandomizedSearchCV(pipeline_rf, param_distributions=param_grid_rf, n_iter=10, cv=3, verbose=1, n_jobs=-1, random_state=random_state)  # Tune Random Forest
random_search_svc = RandomizedSearchCV(pipeline_svc, param_distributions=param_grid_svc, n_iter=10, cv=3, verbose=1, n_jobs=-1, random_state=random_state)  # Tune SVM

# Training models using the best parameters
random_search_lr.fit(X_train, y_train)  # Logistic Regression
random_search_rf.fit(X_train, y_train)  # Random Forest
random_search_svc.fit(X_train, y_train)  # SVM

# Defining a function to evaluate a model on the test set
def evaluate_model(random_search, X_test, y_test):
    """
    Evaluates a trained model on test data and calculates various metrics.
    """
    y_pred = random_search.best_estimator_.predict(X_test)  # Predicts using the best model
    accuracy = accuracy_score(y_test, y_pred)  # Computes accuracy
    precision = precision_score(y_test, y_pred)  # Computes precision
    recall = recall_score(y_test, y_pred)  # Computes recall
    f1 = f1_score(y_test, y_pred)  # Computes F1 score
    return accuracy, precision, recall, f1, y_pred

# Evaluates each model and store results
results = {}
results['Logistic Regression'] = evaluate_model(random_search_lr, X_test, y_test)
results['Random Forest'] = evaluate_model(random_search_rf, X_test, y_test)
results['Support Vector Classifier'] = evaluate_model(random_search_svc, X_test, y_test)

# Identifies and evaluate the best-performing model (Logistic Regression)
print("Best Model: Logistic Regression")
best_model_results = results['Logistic Regression']
print(f"Accuracy: {best_model_results[0]:.4f}, Precision: {best_model_results[1]:.4f}, Recall: {best_model_results[2]:.4f}, F1 Score: {best_model_results[3]:.4f}")

# Displaying classification report for Logistic Regression
y_pred_lr = random_search_lr.best_estimator_.predict(X_test)
print("\nClassification Report for Logistic Regression:")
print(classification_report(y_test, y_pred_lr, target_names=le.classes_))

# Confusion matrix for Logistic Regression
cm = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC curve for Logistic Regression
y_prob_lr = random_search_lr.best_estimator_.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob_lr)
roc_auc = auc(fpr, tpr)
print(f"ROC AUC: {roc_auc:.2f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Logistic Regression')
plt.legend(loc='lower right')
plt.show()

# GUI for spam detection
def predict_message():
    """
    Takes user input, preprocesses it, and predicts whether it's spam or ham.
    """
    user_input = entry_message.get()  # Gets message from user
    if user_input.strip():  # Ensures the input is not empty
        processed_message = preprocess_email(user_input)  # Preprocesses the input
        prediction = random_search_lr.best_estimator_.predict([processed_message])[0]  # Predicts using the best model
        result = "Spam" if prediction == 1 else "Ham"  # Interprets the prediction
        label_result.config(text=f"Prediction: {result}")  # Displays the result
    else:
        label_result.config(text="Please enter a valid message.")

# Creates the GUI window
root = tk.Tk()
root.title("Spam Detector")
root.geometry("400x300")

# Adds widgets to the GUI
# Creates a label widget to display the title of the application
label_title = ttk.Label(root, text="Spam Detector", font=("Arial", 16))
label_title.pack(pady=10)  # Adds padding around the title for better appearance

# Creates a label widget to prompt the user to enter their message
label_message = ttk.Label(root, text="Enter your message:")
label_message.pack(pady=5)  # Adds some spacing between the label and other elements

# Creates an entry widget for the user to input their message
entry_message = ttk.Entry(root, width=50)  # Sets the width of the text box
entry_message.pack(pady=5)  # Adds padding below the entry widget for spacing

# Creates a button widget for the user to trigger the prediction
button_predict = ttk.Button(root, text="Predict", command=predict_message)  # Connects to the prediction function
button_predict.pack(pady=10)  # Adds padding around the button for better spacing

# Creates a label widget to display the result (spam or ham) after prediction
label_result = ttk.Label(root, text="", font=("Arial", 14))  # Empties initially, updates after prediction
label_result.pack(pady=10)  # Adds padding around the result label for appearance

# Runs the GUI application
root.mainloop()
