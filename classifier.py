#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# NLTK data files
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Paths to the dataset files
training_data_file = "train_data.txt"
validation_data_file = "test_data_solution.txt"
test_data_file = "test_data.txt"

# Loading the datasets
train_df = pd.read_csv(training_data_file, delimiter=" ::: ", names=["index", "movie_name", "genre", "description"], engine='python')
validation_df = pd.read_csv(validation_data_file, delimiter=" ::: ", names=["index", "movie_name", "genre", "description"], engine='python')
test_df = pd.read_csv(test_data_file, delimiter=" ::: ", names=["index", "movie_name", "description"], engine='python')

# Combining training and validation data
combined_df = pd.concat([train_df, validation_df])

# Initializing stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to clean text
def clean_text(text):
    # Removing special characters and numbers
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Converting to lowercase
    text = text.lower()
    # Removing stopwords and lemmatize
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words and len(word) > 2)
    return text

# Applying text cleaning to the descriptions
combined_df['cleaned_description'] = combined_df['description'].apply(clean_text)
test_df['cleaned_description'] = test_df['description'].apply(clean_text)

# TF-IDF Vectorization on the cleaned data
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))  # Increased max_features to capture more information
X = tfidf_vectorizer.fit_transform(combined_df['cleaned_description'])
y = combined_df['genre']

# Splitting the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing the Multinomial Naive Bayes classifier
classifier = MultinomialNB(alpha=0.1)  # Decreased alpha to make the model less smooth and more precise

# Training the classifier on the training data
classifier.fit(X_train, y_train)

# Transforming the test data using the trained TF-IDF vectorizer
X_test = tfidf_vectorizer.transform(test_df['cleaned_description'])

# Predicting the genre for the test dataset
y_pred = classifier.predict(X_test)

# Adding the predicted genres to the test dataframe
test_df['predicted_genre'] = y_pred

# Evaluating the model on the validation set (only accuracy)
y_val_pred = classifier.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)

# Printing the accuracy
print(f"Validation Accuracy: {accuracy:.4f}")

# Saving the results
test_df.to_csv("predicted_genres.csv", index=False)

# Function to predict genre for user input
def predict_genre(plot):
    cleaned_plot = clean_text(plot)
    plot_vector = tfidf_vectorizer.transform([cleaned_plot])
    predicted_genre = classifier.predict(plot_vector)[0]
    return predicted_genre

# User interaction for predicting genres
print("\nEnter a movie plot to get the predicted genre (or 'quit' to exit):")
while True:
    user_input = input("Movie Plot: ")
    if user_input.lower() == 'quit':
        break
    else:
        genre = predict_genre(user_input)
        print(f"The predicted genre for the given plot is: {genre}")
