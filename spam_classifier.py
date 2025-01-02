import re  # Import regular expression library for text processing
import string  # Import string library to handle punctuation removal
from sklearn.model_selection import train_test_split, GridSearchCV  # For splitting data into training and testing sets and hyperparameter tuning
from sklearn.feature_extraction.text import TfidfVectorizer  # Converts text into numerical features (TF-IDF representation)
from sklearn.linear_model import LogisticRegression  # Logistic Regression model for classification
from sklearn.ensemble import RandomForestClassifier  # Random Forest classifier
from sklearn.svm import SVC  # Support Vector Classifier for classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc  # Evaluation metrics for model performance
from imblearn.over_sampling import RandomOverSampler  # Used to handle class imbalance by oversampling the minority class
from imblearn.pipeline import Pipeline  # To create a pipeline that chains transformations and classifiers together
import seaborn as sns  # For visualization, especially confusion matrix heatmaps
import matplotlib.pyplot as plt  # For plotting visualizations like confusion matrix and ROC curve
from sklearn.preprocessing import LabelEncoder  # Encodes categorical target labels into numeric values
import nltk  # Natural language processing (NLP) library for text processing tasks
from nltk.stem import WordNetLemmatizer  # For lemmatizing words, reducing them to their base form
from nltk.corpus import stopwords  # To filter out common stopwords (e.g., 'the', 'is') during preprocessing
import pandas as pd  # For data manipulation, reading CSV files, and handling datasets
import tkinter as tk  # GUI components for creating the user interface
from tkinter import messagebox  # For displaying message boxes in the GUI
# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')  # Download tokenizer resources
nltk.download('stopwords')  # Download common stopwords list
nltk.download('wordnet')  # Download lemmatizer resources

# Sets a random seed to ensure reproducibility
random_state = 42

# Load the dataset into a pandas DataFrame
df = pd.read_csv('reduced_dataset_20030228_from_unzipped.csv')  # Dataset containing email data

# Initialize the WordNetLemmatizer and stopwords list
lemmatizer = WordNetLemmatizer()  # Lemmatizer reduces words to base forms (e.g., "running" to "run")
stop_words = set(stopwords.words('english'))  # Stopwords are common words to ignore during text processing

# Preprocessing function for cleaning email text
def preprocess_email(text, remove_punctuation=True, lowercase=True, remove_urls=True, remove_numbers=True, remove_headers=True):
    """
    Preprocesses the email text by:
    - Removing email headers (e.g., 'From', 'To', 'Subject')
    - Replacing numbers with the placeholder 'NUMBER'
    - Removing URLs
    - Converting text to lowercase
    - Removing punctuation
    - Lemmatizing words
    - Removing stopwords

    Args:
        text (str): The email text to preprocess.
        remove_punctuation (bool): Whether to remove punctuation.
        lowercase (bool): Whether to convert text to lowercase.
        remove_urls (bool): Whether to remove URLs.
        remove_numbers (bool): Whether to replace numbers with a placeholder.
        remove_headers (bool): Whether to remove email headers.

    Returns:
        str: Cleaned and preprocessed text.
    """
    if remove_headers:
        # Removes headers like 'From:', 'To:', 'Subject:', 'Date:'
        text = re.sub(r'^(From:|To:|Subject:|Date:)[^\n]*\n', '', text, flags=re.MULTILINE)

    if remove_numbers:
        # Replace all numeric sequences with the word 'NUMBER'
        text = re.sub(r'\d+', 'NUMBER', text)

    if remove_urls:
        # Remove URLs (e.g., 'http://example.com')
        text = re.sub(r'http\S+|www\S+', 'URL', text)

    if lowercase:
        # Convert the entire text to lowercase
        text = text.lower()

    if remove_punctuation:
        # Remove punctuation using string translation
        text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize, lemmatize words, and remove stopwords
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    # Recombine words into a single string
    return ' '.join(words)

# Apply preprocessing to the 'subject' column of the dataset
df['cleaned_text'] = df['subject'].apply(preprocess_email)  # Creates a new column for cleaned text

# Encode the target labels (spam/ham) as numerical values (0 for ham, 1 for spam)
le = LabelEncoder()  # Converts categorical labels to numeric values
df['label'] = le.fit_transform(df['class'])  # Maps 'ham' -> 0 and 'spam' -> 1

# Split the dataset into features (X) and target labels (y)
X = df['cleaned_text']  # Text data after preprocessing
y = df['label']  # Encoded labels

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# Define machine learning pipelines for different models
pipeline_lr = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words='english', binary=True, max_features=10000, ngram_range=(1, 3))),  # TF-IDF conversion
    ('oversample', RandomOverSampler(random_state=random_state)),  # Balance classes using oversampling
    ('classifier', LogisticRegression(solver='liblinear', class_weight='balanced'))  # Logistic Regression classifier
])

pipeline_rf = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words='english', binary=True, max_features=10000, ngram_range=(1, 3))),  # TF-IDF conversion
    ('oversample', RandomOverSampler(random_state=random_state)),  # Balance classes using oversampling
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=random_state))  # Random Forest classifier
])

pipeline_svc = Pipeline([
    # - Converts raw text into numerical features using the Term Frequency-Inverse Document Frequency (TF-IDF) representation.
    # - TF-IDF calculates the importance of each word relative to a document and the entire dataset, reducing the impact of very common words.

    ('vectorizer', TfidfVectorizer(
        stop_words='english',  # Automatically removes common English stopwords (e.g., "the", "is"), which often don't add meaningful information.
        binary=True,           # Represents the presence (1) or absence (0) of each term, rather than counting term frequencies.
        max_features=10000,    # Limits the vocabulary size to the top 10,000 most frequent terms, balancing computational efficiency and model performance.
        ngram_range=(1, 3)     # Considers unigrams (individual words), bigrams (two-word combinations), and trigrams (three-word combinations).
    )),
    ('oversample', RandomOverSampler(random_state=random_state)),  # Balance classes using oversampling
    ('classifier', SVC(class_weight='balanced', probability=True, random_state=random_state))  # SVC classifier
])

# Defines a common parameter grid for all models to tune the TF-IDF vectorizer settings
param_grid = {
    # Max document frequency: Ignore very common words (appear in more than a certain percentage of documents)
    # If a word appears in a large portion of the documents (e.g., 75%, 85%, or 95%), it is likely to be less informative
    # and is excluded from the feature set.
    'vectorizer__max_df': [0.75, 0.85, 0.95],  
    
    # Min document frequency: Include words that appear in at least these many documents
    # Words that appear too infrequently (e.g., in only 1 or 2 documents) are likely to be noise and are excluded.
    # 'min_df=1' allows words appearing at least once in the corpus, while 'min_df=2' excludes those that appear very rarely.
    'vectorizer__min_df': [1, 2],  
    
    # N-gram range: This defines the range of n-grams (sequences of n words) to be considered.
    # (1, 1) = Unigrams (single words), (1, 2) = Unigrams and Bigrams (pairs of consecutive words), 
    # (1, 3) = Unigrams, Bigrams, and Trigrams (triplets of consecutive words).
    # Including n-grams allows the model to capture context and semantic meaning that individual words alone might miss.
    'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)]  
}

# Extend the grid for specific models
param_grid_lr = {**param_grid, 'classifier__C': [0.01, 0.1, 1, 10, 100]}  # Logistic Regression regularization
param_grid_rf = {**param_grid, 'classifier__n_estimators': [100, 200, 500], 'classifier__max_depth': [10, 20, 50]}  # RF trees & depth
param_grid_svc = {**param_grid, 'classifier__C': [0.01, 0.1, 1, 10, 100]}  # SVC regularization

# Perform GridSearchCV to tune hyperparameters for each model
grid_search_lr = GridSearchCV(pipeline_lr, param_grid_lr, cv=3, verbose=1, n_jobs=-1)
grid_search_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=3, verbose=1, n_jobs=-1)
grid_search_svc = GridSearchCV(pipeline_svc, param_grid_svc, cv=3, verbose=1, n_jobs=-1)

# Fit the models using the training data
grid_search_lr.fit(X_train, y_train)
grid_search_rf.fit(X_train, y_train)
grid_search_svc.fit(X_train, y_train)

# Function to evaluate model performance
def evaluate_model(grid_search, X_test, y_test):
    """
    Evaluates the model using test data and returns metrics.

    Args:
        grid_search: GridSearchCV object with the best model.
        X_test: Features for testing.
        y_test: True labels for testing.

    Returns:
        tuple: Accuracy, precision, recall, F1-score, and predictions.
    """
    y_pred = grid_search.best_estimator_.predict(X_test)  # Predictions using the best model

    accuracy = accuracy_score(y_test, y_pred)  # Overall accuracy
    precision = precision_score(y_test, y_pred)  # Precision: True Positives / (True Positives + False Positives)
    recall = recall_score(y_test, y_pred)  # Recall: True Positives / (True Positives + False Negatives)
    f1 = f1_score(y_test, y_pred)  # F1 Score: Harmonic mean of precision and recall

    return accuracy, precision, recall, f1, y_pred

# Evaluate and store results for each model
results = {}
results['Logistic Regression'] = evaluate_model(grid_search_lr, X_test, y_test)
results['Random Forest'] = evaluate_model(grid_search_rf, X_test, y_test)
results['SVC'] = evaluate_model(grid_search_svc, X_test, y_test)

# Display classification reports for each model
for model_name, result in results.items():
    print(f"\nClassification Report for {model_name}:")
    y_pred = result[4]  # Extract predictions
    print(classification_report(y_test, y_pred, target_names=le.classes_))

# Plot confusion matrices
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

# Plot ROC curves
plt.figure(figsize=(10, 8))
for model_name, result in results.items():
    y_pred_prob = grid_search_lr.best_estimator_.predict_proba(X_test)[:, 1]  # Probability of spam class
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# GUI for spam classification
def classify_message():
    """
    Classifies a user-entered message as spam or ham using the best Logistic Regression model.
    """
    message = entry.get()  # Get user input
    cleaned_message = preprocess_email(message)  # Preprocess the input
    model = grid_search_lr.best_estimator_  # Use the best model
    prediction = model.predict([cleaned_message])
    if prediction == 1:
        messagebox.showinfo("Prediction", "This message is Spam!")
    else:
        messagebox.showinfo("Prediction", "This message is Ham!")

# Build the GUI
window = tk.Tk()
window.title("Spam Classifier")

entry = tk.Entry(window, width=50)  # Input field
entry.pack(pady=10)

button = tk.Button(window, text="Classify", command=classify_message)  # Classify button
button.pack(pady=5)

window.mainloop()
