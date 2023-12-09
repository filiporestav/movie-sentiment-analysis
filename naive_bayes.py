import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# If not already downloaded, download the stopwords dataset
#nltk.download('stopwords')
# Download the punkt tokenizer data
#nltk.download('punkt')

# Function to load test and train data
def load_data(directory):
    reviews = []
    labels = []
    for label in ['pos', 'neg']:
        folder_path = os.path.join(directory, label)
        for filename in os.listdir(folder_path):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                review = file.read()
                reviews.append(review)
                labels.append(label)
    return pd.DataFrame({'review': reviews, 'label': labels})

def load_word_exp_rating():
    dictionary = {} # Store the {word: average word polarity of the word}
    vocab = 'data/imdb.vocab' # The vocabulary
    word_polarity = 'data/imdbEr.txt' # The average word polarity

    # Open both files
    with open(vocab, 'r', encoding='utf-8') as file1, open(word_polarity, 'r', encoding='utf-8') as file2:
        # Iterate over the lines of both files simultaneously
        for line1, line2 in zip(file1, file2):
            # Process each line as needed
            word = line1.strip()  # Assuming each line in file1 contains a single word
            avg_word_polarity = line2.strip()  # Assuming each line in file2 contains the corresponding value
            dictionary[word] = avg_word_polarity
    return dictionary


def pre_process(text):
    words = word_tokenize(text) # Get words as a list
    filtered_words = [word for word in words if word.lower() in unwanted_words]
    return " ".join(filtered_words)

# Function to save the classifier and vectorizer
def save_model(classifier, vectorizer, model_filename, vectorizer_filename):
    joblib.dump(classifier, model_filename)
    joblib.dump(vectorizer, vectorizer_filename)

def predict_label(text):
    return model.predict(vectorizer.transform([text]))

average_word_polarity = load_word_exp_rating() # Dictionary with avg word polarity

# Load data
train_data = load_data('data/train')
test_data = load_data('data/test')

# Set a list with unwanted forms
unwanted_words = set(stopwords.words("english"))
other_words_to_remove = {"movie", "br", "film", "really", "just", "having", "doing"}
unwanted_words.update(other_words_to_remove)

# Text preprocessing using TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words=unwanted_words, lowercase=True)
X_train = vectorizer.fit_transform(train_data['cleaned_review'])
y_train = train_data['label']

X_test = vectorizer.transform(test_data['cleaned_review'])
y_test = test_data['label']

# Build a Sentiment Classification Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save the model and vectorizer
save_model(model, vectorizer, 'sentiment_model.joblib', 'tfidf_vectorizer.joblib')

# Get feature names from the TfidfVectorizer
feature_names = np.array(vectorizer.get_feature_names_out())

# Get the most informative features for each class
top_positive_features = feature_names[np.argsort(model.feature_log_prob_[1])[::-1][:10]]
top_negative_features = feature_names[np.argsort(model.feature_log_prob_[0])[::-1][:10]]

print("Top positive features:", top_positive_features)
print("Top negative features:", top_negative_features)

# Evaluate the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Validation Accuracy: {accuracy:.2f}')
print('Validation Classification Report:')
print(classification_report(y_test, y_pred))

print(predict_label("This was a really good movie"))

# Here we can use loaded_model and loaded_vectorizer for predictions without retraining.
# loaded_model = joblib.load('sentiment_model.joblib')
# loaded_vectorizer = joblib.load('tfidf_vectorizer.joblib')