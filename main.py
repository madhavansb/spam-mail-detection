import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Download NLTK stopwords (if not already downloaded)
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('/spam_ham_dataset.csv')

# Debug: Print available columns
print("Available columns in dataset:", df.columns)

# **Find the correct column for text messages**
# Common names for text columns in spam datasets: "message", "text", "email"
possible_text_columns = ["text", "message", "email", "content"]
text_column = None

for col in possible_text_columns:
    if col in df.columns:
        text_column = col
        break

if text_column is None:
    raise KeyError("Could not find a column with email/text messages. Check your dataset.")

# **Ensure the column has valid data**
df[text_column] = df[text_column].astype(str).str.replace('\r\n', ' ', regex=False)

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stopwords_set = set(stopwords.words('english'))

# Preprocess text data
corpus = []
for text in df[text_column]:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stopwords_set]  # Stem and remove stopwords
    cleaned_text = ' '.join(words)
    corpus.append(cleaned_text)

# Ensure corpus is not empty
if not any(corpus):
    raise ValueError("Processed corpus is empty. Ensure valid data in the text column.")

# Convert text to numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus).toarray()

# **Find the correct label column**
label_column = None
possible_label_columns = ["label", "label_num", "category", "is_spam"]
for col in possible_label_columns:
    if col in df.columns:
        label_column = col
        break

if label_column is None:
    raise KeyError("Could not find a label column indicating spam or not. Check your dataset.")

y = df[label_column].values  # Convert to NumPy array

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
clf = RandomForestClassifier(n_jobs=-1, random_state=42)
clf.fit(X_train, y_train)

# **Function to classify user input with spam probability**
def classify_email(email_text):
    """Preprocesses user input and predicts the spam probability."""
    email_text = email_text.lower().translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = email_text.split()
    words = [stemmer.stem(word) for word in words if word not in stopwords_set]
    cleaned_text = ' '.join(words)

    # Transform text using the trained vectorizer
    X_mail = vectorizer.transform([cleaned_text])

    # Get prediction probability
    spam_probability = clf.predict_proba(X_mail)[0][1]  # Probability of spam (class 1)
    return spam_probability * 100  # Convert to percentage

# **Get user input and classify**
user_email = input("Enter an email message to classify: ")
spam_percent = classify_email(user_email)
print(f"\nSpam Probability: {spam_percent:.2f}%")
