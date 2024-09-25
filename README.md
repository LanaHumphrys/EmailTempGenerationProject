"Email Template Generation Project - aims to generate email templates using Natural Language Processing (NLP) techniques."

# Load Libraries required for data processing, analysis, and visualization.

import os
import pandas as pd # "pandas" and "numpy" are used for data manipulation.
import re # "re" is used for regular expressions to clean text data.
import spacy # "spacy" is used for NLP tasks such as named entity recognition. 
# It will help processing large volumes of text and tuning configurations to match specific use 
# cases in a way that provides better accuracy
from sklearn.model_selection import train_test_split # sklearn libraries are used for machine learning tasks
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt # matplotlib and plotly are used for data visualization.
import plotly.express as px
from transformers import pipeline # transformers is used for implementing pre-trained models
from docx import Document ## docx is used to extract text from .docx files.
from textblob import TextBlob # textblob is used for sentiment analysis
from nltk import ngrams ## `nltk` is used for natural language processing tasks like tokenization and stopword removal
from wordcloud import WordCloud
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk


# 1. Data Collection
# Collect the data from provided documents (e.g., .docx files) to extract email content for further processing.
from docx import Document

# Function to extract text from docx files
def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

# Example usage
file_path = 'path_to_your_docx_file'
text = extract_text_from_docx(file_path)
print(text)


Explanation steps:
- This section focuses on collecting data from .docx files.
- The "extract_text_from_docx" function reads a .docx file and extracts the text from each paragraph.
- The extracted text is then joined into a single string with newline characters separating the paragraphs.
- The example usage demonstrates how to use this function to extract text from a specified .docx file.

# 2. Data Preprocessing

# python
## **2. Data Preprocessing
# Clean the email text by removing unnecessary information, standardizing formats, and preparing the data for analysis.
import re

# Preprocess the extracted text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

# Example usage
cleaned_text = preprocess_text(text)
print(cleaned_text)


# Explanation:
- This section is dedicated to cleaning the extracted text data.
- The `preprocess_text` function performs several preprocessing steps:
  - Converts the text to lowercase to ensure uniformity.
  - Removes numbers using regular expressions.
  - Removes extra whitespace and trims the text.
- The example usage shows how to apply this function to the extracted text.

# 3. Data Annotation

# python
# 3. Data Annotation
# Label the preprocessed emails with metadata such as structure, tone, and purpose for training the NLP model.
# Example of adding a manual label (e.g., tone, purpose)
emails = [{'email': cleaned_text, 'category': 'Meeting Request', 'tone': 'Formal'}]

# Display the annotated email
print(emails)

# Explanation Notes:
- This section involves annotating the preprocessed emails with metadata.
- The example demonstrates how to manually label an email with categories such as 'Meeting Request' and tone such as 'Formal'.
- This metadata will be useful for training the NLP model to understand the context and purpose of the emails.

# 4. Exploratory Data Analysis (EDA)

# python
# 4. Exploratory Data Analysis (EDA)
# Explore patterns, word frequencies, and email structures in the annotated data to guide model selection.
# Basic Descriptive Statistics
# Calculate number of words and characters per email
emails_df['word_count'] = emails_df['cleaned_email_body'].apply(lambda x: len(word_tokenize(x)))
emails_df['char_count'] = emails_df['cleaned_email_body'].apply(lambda x: len(x))

# Display basic statistics
print(emails_df[['word_count', 'char_count']].describe())

# Sentiment Analysis
from textblob import TextBlob

# Sentiment Analysis
emails_df['sentiment'] = emails_df['cleaned_email_body'].apply(lambda x: TextBlob(x).sentiment.polarity)
emails_df['subjectivity'] = emails_df['cleaned_email_body'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

# Display sentiment distribution
print(emails_df[['sentiment', 'subjectivity']].describe())

# Bigram and Trigram Analysis
from nltk import ngrams

def generate_ngrams(text, n):
    tokens = word_tokenize(text)
    n_grams = list(ngrams(tokens, n))
    return n_grams

# Generate bigrams
emails_df['bigrams'] = emails_df['cleaned_email_body'].apply(lambda x: generate_ngrams(x, 2))

# Display most common bigrams
bigram_freq = Counter([bigram for sublist in emails_df['bigrams'] for bigram in sublist])
print(bigram_freq.most_common(10))

# Word Cloud for Visualization
from wordcloud import WordCloud

# Create a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(tokens))

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Named Entity Recognition (NER)
import spacy

# Load SpaCy's pre-trained NER model
nlp = spacy.load('en_core_web_sm')

# Extract named entities from the text
def extract_named_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Apply NER to the email content
emails_df['entities'] = emails_df['cleaned_email_body'].apply(extract_named_entities)

# Display the extracted entities
print(emails_df[['cleaned_email_body', 'entities']].head())

# Keyword/Topic Extraction
from sklearn.feature_extraction.text import TfidfVectorizer

# Compute TF-IDF for the email dataset
tfidf = TfidfVectorizer(stop_words='english', max_features=100)
tfidf_matrix = tfidf.fit_transform(emails_df['cleaned_email_body'])

# Display top TF-IDF words
feature_names = tfidf.get_feature_names_out()
print("Top 10 TF-IDF words:", feature_names[:10])

# Email Category Distribution
# Assuming the dataset has a 'category' column
category_counts = emails_df['category'].value_counts()

# Plot category distribution
category_counts.plot(kind='bar', title='Email Category Distribution')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()

from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt

# Download stopwords for tokenization
nltk.download('punkt')
nltk.download('stopwords')

# Tokenize and remove stopwords
def tokenize_and_remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return filtered_tokens

# Example usage
tokens = tokenize_and_remove_stopwords(cleaned_text)
word_freq = Counter(tokens)

# Plot word frequencies
common_words = word_freq.most_common(10)
words, counts = zip(*common_words)
plt.bar(words, counts)
plt.title('Top 10 Most Common Words')
plt.show()


# This section involves exploring the annotated data to uncover patterns and insights:
- Basic Descriptive Statistics**: Calculates and displays the number of words and characters per email.
- Sentiment Analysis**: Uses `TextBlob` to analyze the sentiment and subjectivity of the email content.
- Bigram and Trigram Analysis**: Generates and displays the most common bigrams (pairs of words) in the email content.
- Word Cloud for Visualization**: Creates and displays a word cloud to visualize the most common words in the email content.
- Named Entity Recognition (NER)**: Uses `spaCy` to extract named entities from the email content.
- Keyword/Topic Extraction**: Computes TF-IDF (Term Frequency-Inverse Document Frequency) to identify the most important words in the email content.
- Email Category Distribution**: Plots the distribution of email categories.
- Tokenize and Remove Stopwords**: Tokenizes the email content and removes stopwords, then plots the most common words.

# 5. Model Selection and Design

markdown
# 5. Model Selection and Design**
# Choose an appropriate NLP model
# Begin by considering pre-trained models that are well-suited for text generation and fine-tuning.
# Configure it for email template generation.


# Explanation notes:
- This section involves selecting and designing the NLP model for email template generation.
- It suggests considering pre-trained models that are suitable for text generation and fine-tuning them for the specific task of generating email templates.

# 6. Model Training

markdown
# 6. Model Training
# Train the NLP model using the annotated data to generate context-specific email templates.


# Explanation note:
- This section focuses on training the selected NLP model using the annotated data.
- The goal is to teach the model to generate context-specific email templates based on the training data.

# 7. Implement Reinforcement Learning**

markdown
# 7. Implement Reinforcement Learning**
# Implement a feedback mechanism to improve the model's performance based on user feedback during email template generation.


# Explanation:
- This section involves implementing reinforcement learning to continuously improve the model's performance.
- A feedback mechanism is used to gather user feedback on the generated email templates, which is then used to refine and enhance the model.
