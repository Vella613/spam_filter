import re  # For regular expressions to process text
import string  # To handle punctuation removal
from sklearn.model_selection import train_test_split, GridSearchCV  # For splitting data and hyperparameter tuning
from sklearn.feature_extraction.text import TfidfVectorizer  # To convert text to numeric features
from sklearn.linear_model import LogisticRegression  # Logistic Regression model
from sklearn.ensemble import RandomForestClassifier  # Random Forest classifier
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc  # For model evaluation
from imblearn.over_sampling import RandomOverSampler  # For handling class imbalance by oversampling the minority class
from imblearn.pipeline import Pipeline  # To create a pipeline of transformations and models
import seaborn as sns  # For visualization, especially confusion matrix heatmaps
import matplotlib.pyplot as plt  # For plotting visualizations
from sklearn.preprocessing import LabelEncoder  # To encode target labels (spam/ham) as numeric values
import nltk  # Natural language processing (NLP) library
from nltk.stem import WordNetLemmatizer  # For word lemmatization
from nltk.corpus import stopwords  # To filter out common words (e.g., 'the', 'is') during preprocessing
import pandas as pd  # For data manipulation
import tkinter as tk  # For GUI components
from tkinter import messagebox  # For showing messages in the GUI

# Ensure NLTK resources are downloaded (for lemmatizer and stopwords)
nltk.download('punkt')  # Tokenizer resources
nltk.download('stopwords')  # Stopword list
nltk.download('wordnet')  # Lemmatization resources

# Set random seed for reproducibility of results
random_state = 42

# Load the dataset into a pandas DataFrame
df = pd.read_csv('reduced_dataset_20030228_from_unzipped.csv')

# Initialize the lemmatizer and load English stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))  # A set of common words to remove from the text

# Define a preprocessing function for email text
def preprocess_email(text, remove_punctuation=True, lowercase=True, remove_urls=True, remove_numbers=True, remove_headers=True):
    # Optional: Remove email headers such as From, To, Subject, Date
    if remove_headers:
        text = re.sub(r'^(From:|To:|Subject:|Date:)[^\n]*\n', '', text, flags=re.MULTILINE)
    
    # Optional: Remove all digits and replace them with the word 'NUMBER'
    if remove_numbers:
        text = re.sub(r'\d+', 'NUMBER', text)
    
    # Optional: Remove URLs from the text
    if remove_urls:
        text = re.sub(r'http\S+|www\S+', 'URL', text)
    
    # Optional: Convert the text to lowercase
    if lowercase:
        text = text.lower()
    
    # Optional: Remove punctuation (e.g., commas, periods, etc.)
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Split text into words and lemmatize each word (i.e., reduce words to their base form)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    # Join the lemmatized words back into a single string
    return ' '.join(words)

# Apply the preprocessing function to the 'subject' column of the dataset
df['cleaned_text'] = df['subject'].apply(preprocess_email)

# Encode the target labels ('class' column) as numerical values (0 or 1)
le = LabelEncoder()
df['label'] = le.fit_transform(df['class'])

# Split the dataset into features (X) and target labels (y)
X = df['cleaned_text']
y = df['label']

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# Define the machine learning pipeline for Logistic Regression
pipeline_lr = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words='english', binary=True, max_features=10000, ngram_range=(1, 3))),  # Convert text to features using TF-IDF
    ('oversample', RandomOverSampler(random_state=random_state)),  # Apply oversampling to balance classes
    ('classifier', LogisticRegression(solver='liblinear', class_weight='balanced'))  # Logistic Regression model
])

# Define the machine learning pipeline for Random Forest
pipeline_rf = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words='english', binary=True, max_features=10000, ngram_range=(1, 3))),  # Convert text to features using TF-IDF
    ('oversample', RandomOverSampler(random_state=random_state)),  # Apply oversampling to balance classes
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=random_state))  # Random Forest model
])

# Define the machine learning pipeline for Support Vector Classifier
pipeline_svc = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words='english', binary=True, max_features=10000, ngram_range=(1, 3))),  # Convert text to features using TF-IDF
    ('oversample', RandomOverSampler(random_state=random_state)),  # Apply oversampling to balance classes
    ('classifier', SVC(class_weight='balanced', probability=True, random_state=random_state))  # SVC model
])

# Define a common parameter grid for all models to tune the TF-IDF vectorizer settings
param_grid = {
    'vectorizer__max_df': [0.75, 0.85, 0.95],  # Maximum document frequency: threshold for excluding rare words
    'vectorizer__min_df': [1, 2],  # Minimum document frequency: threshold for including words
    'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)]  # Unigrams, bigrams, trigrams (combinations of adjacent words)
}

# Define specific hyperparameter grids for each classifier
param_grid_lr = {**param_grid, 'classifier__C': [0.01, 0.1, 1, 10, 100]}  # Regularization parameter 'C' for Logistic Regression
param_grid_rf = {**param_grid, 'classifier__n_estimators': [100, 200, 500], 'classifier__max_depth': [10, 20, 50]}  # Parameters for Random Forest
param_grid_svc = {**param_grid, 'classifier__C': [0.01, 0.1, 1, 10, 100]}  # Regularization parameter 'C' for SVC

# Perform GridSearchCV for hyperparameter tuning on each model
grid_search_lr = GridSearchCV(pipeline_lr, param_grid_lr, cv=3, verbose=1, n_jobs=-1)
grid_search_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=3, verbose=1, n_jobs=-1)
grid_search_svc = GridSearchCV(pipeline_svc, param_grid_svc, cv=3, verbose=1, n_jobs=-1)

# Fit the models using the training data
grid_search_lr.fit(X_train, y_train)
grid_search_rf.fit(X_train, y_train)
grid_search_svc.fit(X_train, y_train)

# Define a function to evaluate the model's performance and return classification metrics
def evaluate_model(grid_search, X_test, y_test):
    y_pred = grid_search.best_estimator_.predict(X_test)  # Get predictions using the best model
    accuracy = accuracy_score(y_test, y_pred)  # Accuracy score
    precision = precision_score(y_test, y_pred)  # Precision score
    recall = recall_score(y_test, y_pred)  # Recall score
    f1 = f1_score(y_test, y_pred)  # F1 score (harmonic mean of precision and recall)
    return accuracy, precision, recall, f1, y_pred

# Store the results for each model for later comparison
results = {}

# Evaluate the Logistic Regression model
results['Logistic Regression'] = evaluate_model(grid_search_lr, X_test, y_test)

# Evaluate the Random Forest model
results['Random Forest'] = evaluate_model(grid_search_rf, X_test, y_test)

# Evaluate the SVC model
results['SVC'] = evaluate_model(grid_search_svc, X_test, y_test)

# Print classification reports for all models
for model_name, result in results.items():
    print(f"\nClassification Report for {model_name}:")
    y_pred = result[4]  # Get the predictions for this model
    print(classification_report(y_test, y_pred, target_names=le.classes_))  # Detailed classification metrics

# Plot confusion matrices for each model
plt.figure(figsize=(15, 5))

for i, (model_name, result) in enumerate(results.items(), 1):
    y_pred = result[4]  # Get the predictions
    cm = confusion_matrix(y_test, y_pred)  # Calculate confusion matrix
    
    # Plot each confusion matrix as a heatmap
    plt.subplot(1, 3, i)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

plt.tight_layout()
plt.show()

# Plot ROC curve for each model
plt.figure(figsize=(10, 8))

for model_name, result in results.items():
    y_pred_prob = grid_search_lr.best_estimator_.predict_proba(X_test)[:, 1]  # Get the probabilities for positive class (Spam)
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)  # Compute the ROC curve
    auc_score = auc(fpr, tpr)  # Compute the AUC score
    
    # Plot ROC curve
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Plot the baseline (random classifier)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# GUI Setup to type message and classify
def classify_message():
    # Retrieve the message typed by the user
    message = entry.get()
    
    # Preprocess the message
    cleaned_message = preprocess_email(message)
    
    # Predict using the best model (Logistic Regression as an example)
    model = grid_search_lr.best_estimator_
    prediction = model.predict([cleaned_message])
    
    # Show the result in a message box
    if prediction == 1:
        messagebox.showinfo("Prediction", "This message is Spam!")
    else:
        messagebox.showinfo("Prediction", "This message is Ham!")

# Create a GUI window
window = tk.Tk()
window.title("Spam Classifier")

# Create an entry box for message input
entry = tk.Entry(window, width=50)
entry.pack(pady=10)

# Create a button to classify the message
button = tk.Button(window, text="Classify", command=classify_message)
button.pack(pady=5)

# Run the GUI loop
window.mainloop()
