import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Download necessary NLTK corpora silently
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

class NLPTextPipeline:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(max_features=5000)
    
    def clean_text(self, text):
        """Standard NLP pipeline: Lowercase, remove punct/numbers, remove stopwords, lemmatize."""
        if not isinstance(text, str):
            return ""
        # Lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove punctuation and numbers
        text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Tokenize (simple split is faster, but we'll use NLTK logic through split for simplicity)
        words = text.split()
        
        # Remove Stopwords and Lemmatize
        cleaned_words = [
            self.lemmatizer.lemmatize(word) 
            for word in words 
            if word not in self.stop_words and len(word) > 1
        ]
        
        return " ".join(cleaned_words)
    
    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)
        
    def transform(self, texts):
        return self.vectorizer.transform(texts)
        
def evaluate_model(y_test, y_pred, model_name, class_names, project_name):
    """Generates classification report and confusion matrix plot."""
    os.makedirs('results', exist_ok=True)
    
    print(f"\n--- {model_name} Results ({project_name}) ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    if len(class_names) == 2:
        # Binary classification
        print(f"Precision: {precision_score(y_test, y_pred, average='binary'):.4f}")
        print(f"Recall:    {recall_score(y_test, y_pred, average='binary'):.4f}")
        print(f"F1-Score:  {f1_score(y_test, y_pred, average='binary'):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix: {model_name}\n({project_name})')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"results/conf_matrix_{project_name.lower().replace(' ', '_')}_{model_name.replace(' ', '_').lower()}.png")
    plt.close()
