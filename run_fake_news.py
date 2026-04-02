import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from nlp_pipeline import NLPTextPipeline, evaluate_model

DATA_URL = "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv"
DATA_FILE = "fake_or_real_news.csv"

def download_and_load_data():
    """Downloads the Fake News dataset from an open source Github repo."""
    os.makedirs('data', exist_ok=True)
    file_path = os.path.join('data', DATA_FILE)
    
    if not os.path.exists(file_path):
        print("Downloading Fake News dataset (~30MB) - this might take a minute...")
        response = requests.get(DATA_URL)
        with open(file_path, 'wb') as f:
            f.write(response.content)
            
    print("Loading dataset...")
    # The dataset has an index column, 'title', 'text', 'label' ("FAKE" or "REAL")
    df = pd.read_csv(file_path, index_col=0)
    return df

def plot_top_features(vectorizer, model, top_n=20):
    """Visualizes the top important words/features for Logistic Regression."""
    print("Extracting feature importances...")
    feature_names = vectorizer.get_feature_names_out()
    
    if hasattr(model, 'coef_'):
        # For Logistic Regression, positive coefs mean Fake (if Fake=1)
        # We'll plot absolute importance to see which words matter most overall
        coefs_with_fns = sorted(zip(model.coef_[0], feature_names))
        
        # Most real words (negative coefs if Fake=1 and Real=0)
        top_real = coefs_with_fns[:top_n//2]
        # Most fake words (positive coefs)
        top_fake = coefs_with_fns[-top_n//2:]
        
        top_coefs = [c[0] for c in top_real + top_fake]
        top_words = [c[1] for c in top_real + top_fake]
        
        plt.figure(figsize=(10, 6))
        # Color red for fake (positive), blue for real (negative)
        colors = ['red' if c > 0 else 'blue' for c in top_coefs]
        sns.barplot(x=top_coefs, y=top_words, palette=colors)
        plt.title('Top Most Important Words (Red = Fake, Blue = Real)')
        plt.xlabel('Coefficient Value')
        plt.tight_layout()
        plt.savefig(f'results/fake_news_feature_importance.png')
        plt.close()
        
    elif hasattr(model, 'feature_importances_'):
        # For Random Forest
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], palette='magma')
        plt.title('Top Important Words (Random Forest)')
        plt.tight_layout()
        plt.savefig(f'results/fake_news_rf_feature_importance.png')
        plt.close()

def main():
    print("=== Fake News Detection System ===")
    df = download_and_load_data()
    print(f"Dataset Shape: {df.shape}")
    
    # Preprocessing
    pipeline = NLPTextPipeline()
    print("Cleaning text data. Due to long articles, this will take some time...")
    
    # To save time for demonstration purposes if dataset is huge, we concat title and text
    df['full_content'] = df['title'] + " " + df['text']
    
    X_clean = df['full_content'].apply(pipeline.clean_text)
    
    # Target: 0 for REAL, 1 for FAKE
    y = df['label'].map({'REAL': 0, 'FAKE': 1})
    
    # TF-IDF Vectorization
    print("Building TF-IDF Matrix...")
    X_vect = pipeline.vectorizer.fit_transform(X_clean)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.25, random_state=42, stratify=y)
    
    # Train Models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "XGBoost (GPU)": XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', tree_method='hist', device='cuda')
    }
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        evaluate_model(y_test, y_pred, name, ["Real", "Fake"], "Fake News Detector")
        
        # Plot features for both
        plot_top_features(pipeline.vectorizer, model, top_n=20)
        
    print("\nProcessing complete! Check 'results' folder for confusion matrices and feature importance visualization.")

if __name__ == "__main__":
    main()
