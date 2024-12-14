import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import dump
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from tqdm import tqdm
import time

# Download NLTK resources
nltk.download('stopwords')

# Initialize tqdm for progress bar
tqdm.pandas()

# Start timer
start_time = time.time()

# Define text cleaning function
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    words = text.split()  # Tokenize words
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Load datasets
train_data = pd.read_csv(r"C:\Users\abdoh\Downloads\Sentiment Analysis\train.csv")
test_data = pd.read_csv(r"C:\Users\abdoh\Downloads\Sentiment Analysis\test.csv")

# Check if cleaned data exists
if 'Cleaned_Review' not in train_data.columns:
    print("Cleaning training data...")
    train_data['Cleaned_Review'] = train_data['Review'].progress_apply(clean_text)

if 'Cleaned_Review' not in test_data.columns:
    print("Cleaning test data...")
    test_data['Cleaned_Review'] = test_data['Review'].progress_apply(clean_text)

# Bag of Words Transformation
vectorizer = CountVectorizer(max_features=5000)

print("Fitting CountVectorizer to training data...")
X_train = vectorizer.fit_transform(train_data['Cleaned_Review'])

print("Transforming test data with the fitted vectorizer...")
X_test = vectorizer.transform(test_data['Cleaned_Review'])

# Train Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)

print("Training Logistic Regression model...")
log_reg.fit(X_train, train_data['Label'])

# Make predictions
print("Making predictions on the test data...")
y_pred = log_reg.predict(X_test)

# Evaluate the model
print("\nModel Evaluation Metrics:")

# Accuracy
accuracy = accuracy_score(test_data['Label'], y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Classification report
classification_rep = classification_report(test_data['Label'], y_pred, target_names=['Negative', 'Positive'])
print("\nClassification Report:")
print(classification_rep)

# Confusion matrix
print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(test_data['Label'], y_pred)
print(conf_matrix)

# Save the model for future use
dump(log_reg, "logistic_regression_model.joblib")
print("\nModel saved as 'logistic_regression_model.joblib'.")

# Save evaluation results
with open("evaluation_results.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_rep + "\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(conf_matrix) + "\n")

print("\nEvaluation results saved as 'evaluation_results.txt'.")

# Timer end
end_time = time.time()
print(f"Total processing time: {end_time - start_time:.2f} seconds")
