import os
import requests
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from nlp_pipeline import NLPTextPipeline, evaluate_model

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATA_ZIP = "smsspamcollection.zip"
DATA_FILE = "SMSSpamCollection"

def download_and_load_data():
    """Downloads the UCI SMS Spam Collection dataset."""
    os.makedirs('data', exist_ok=True)
    zip_path = os.path.join('data', DATA_ZIP)
    file_path = os.path.join('data', DATA_FILE)
    
    if not os.path.exists(file_path):
        print("Downloading SMS Spam Collection dataset...")
        response = requests.get(DATA_URL)
        with open(zip_path, 'wb') as f:
            f.write(response.content)
            
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('data')
            
    print("Loading dataset...")
    # The file is a tab-separated value file: label \t message
    df = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'message'])
    return df

def main():
    print("=== Spam / Phishing Classifier ===")
    df = download_and_load_data()
    print(f"Dataset Shape: {df.shape}")
    
    # Preprocessing
    pipeline = NLPTextPipeline()
    print("Cleaning text data (this may take a moment)...")
    # Using pandas apply
    X_clean = df['message'].apply(pipeline.clean_text)
    
    # Target: 0 for ham (normal), 1 for spam
    y = df['label'].map({'ham': 0, 'spam': 1})
    
    # TF-IDF Vectorization
    X_vect = pipeline.vectorizer.fit_transform(X_clean)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.25, random_state=42, stratify=y)
    
    # Train Models
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "XGBoost (GPU)": XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', tree_method='hist', device='cuda')
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        evaluate_model(y_test, y_pred, name, ["Ham", "Spam"], "Spam Classifier")
        trained_models[name] = model

    print("\n--- Sample Predictions ---")
    sample_texts = [
        "Hey man, are we still on for the game tonight?",
        "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net"
    ]
    
    # Process samples
    clean_samples = [pipeline.clean_text(s) for s in sample_texts]
    vect_samples = pipeline.vectorizer.transform(clean_samples)
    
    best_model = trained_models["Naive Bayes"]
    preds = best_model.predict(vect_samples)
    
    for text, pred in zip(sample_texts, preds):
        label = "Spam" if pred == 1 else "Ham/Normal"
        print(f"Message: {text}")
        print(f"Prediction: -> {label}\n")
        
if __name__ == "__main__":
    main()
