# Amazon Sentiment Analysis Using Logistic Regression

This project uses logistic regression to classify Amazon product reviews as Positive or Negative. It involves preprocessing textual data, feature engineering using Bag-of-Words, and model training to achieve high accuracy.

## Table of Contents
- [Introduction](#introduction)
- [Dataset Description](#dataset-description)
- [Methodology](#methodology)
- [Results and Visualizations](#results-and-visualizations)
- [How to Run](#how-to-run)
- [Files in the Repository](#files-in-the-repository)
- [Future Work and Recommendations](#future-work-and-recommendations)
- [Conclusion](#conclusion)

## Introduction
The objective of this project is to build a machine learning model to classify customer sentiment from Amazon product reviews as either Positive or Negative. This enables businesses to analyze feedback effectively.

This project makes use of:
- Programming Language: Python
- Libraries: NLTK, scikit-learn, pandas, tqdm, joblib, matplotlib

## Dataset Description
- Training Dataset: 3.6 million reviews (50% Positive, 50% Negative)
- Test Dataset: 400,000 reviews (50% Positive, 50% Negative)
- Source: Amazon product reviews

## Methodology
### 1. Data Preprocessing
- Removed non-alphabetic characters
- Converted text to lowercase
- Removed stopwords (using NLTK)
- Applied stemming (PorterStemmer)


### Visualizations
- Feature Engineering: Word Frequencies
![Feature_Engineering](https://github.com/user-attachments/assets/e88312e7-ef63-44b8-8a87-2563591d078a)


- Confusion Matrix
![Confusion_Matrix](https://github.com/user-attachments/assets/79d7b02d-1e89-4336-85b3-4c982cc2f096)

- Class Distribution in Training Data
![Training_Data_Distribution](https://github.com/user-attachments/assets/2d485750-72e0-43d1-8045-040cb56a55a2)

- Class Distribution in Test Data
![Test_Data_Distribution](https://github.com/user-attachments/assets/e0e753eb-1e39-4c0c-8eee-a8debaa5bb76)

## How to Run
### Prerequisites
- Python 3.8+
- Install required libraries:
```bash
pip install -r requirements.txt

### 2. Feature Engineering
- Used Bag-of-Words (CountVectorizer)
- Limited vocabulary to 5000 most frequent words

### 3. Model Training
- Trained a Logistic Regression model on the transformed features

### 4. Evaluation
- Accuracy
- Classification Report (Precision, Recall, F1-score)
- Confusion Matrix

## Results and Visualizations
### Key Metrics
- Accuracy: 88.57%
```

## Files in the Repository
Sentiment_Analysis.py: Source code for training and evaluating the model.

logistic_regression_model.joblib: Saved logistic regression model.

evaluation_results.txt: Evaluation metrics including accuracy, classification report, and confusion matrix.

## Visualizations:
Feature_Engineering.png

Confusion_Matrix.png

Training_Data_Distribution.png

Test_Data_Distribution.png

## Future Work and Recommendations

Explore advanced text representations (e.g., TF-IDF, Word2Vec, GloVe).

Tune logistic regression hyperparameters using grid search or random search.

Compare with other models like Naive Bayes, SVM, or deep learning models (LSTMs, BERT).

Use explainability tools like SHAP or LIME to interpret model predictions.

Deploy the model as a web service for real-time sentiment analysis.

## Conclusion

This logistic regression model achieves high accuracy (88.57%) in classifying Amazon product reviews. Its balanced performance across Positive and Negative sentiment classes demonstrates its practical effectiveness. This scalable approach is well-suited for analyzing large text datasets.
