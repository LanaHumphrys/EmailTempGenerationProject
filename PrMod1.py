import os
import re
import spacy
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from transformers import pipeline as transformers_pipeline
from autocorrect import Speller

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Initialize spell checker
spell = Speller()

# Function to remove personal information
def remove_personal_info(text):
    doc = nlp(text)
    return ' '.join([token.text for token in doc if not token.ent_type_ in ['PERSON', 'EMAIL', 'PHONE', 'GPE']])

# Function to correct typos
def correct_typos(text):
    return spell(text)

# Function to standardize format
def standardize_format(text):
    return text.lower()

# Load data
data_dir = 'c:\\project1'
documents = []
labels = []

for filename in os.listdir(data_dir):
    if filename.endswith('.txt'):
        with open(os.path.join(data_dir, filename), 'r') as file:
            text = file.read()
            text = remove_personal_info(text)
            text = correct_typos(text)
            text = standardize_format(text)
            documents.append(text)
            # Assuming labels are in the filename, e.g., 'meeting_1.txt'
            labels.append(filename.split('_'))

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(documents, labels, test_size=0.2, random_state=42)

# Create a pipeline for classification
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# Train the model
pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = pipeline.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()