import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import torch
from transformers import DistilBertTokenizer, DistilBertModel

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device for BERT extraction: {device}")

# Load Pre-trained Tokenizer and Model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
bert_model.eval()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data", "fake_or_real_news.csv")

def extract_bert_embeddings(text_list, batch_size=32):
    """Extracts the [CLS] token embedding from DistilBERT."""
    all_embeddings = []
    
    # We will process in batches to not blow up GPU RAM
    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i:i+batch_size]
        
        # Tokenize and pad to max length 512
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = bert_model(**inputs)
            # Take the CLS token representing the whole sentence (index 0)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
        all_embeddings.extend(cls_embeddings)
        print(f"Processed {min(i+batch_size, len(text_list))}/{len(text_list)} texts...")
        
    return np.array(all_embeddings)

def main():
    if not os.path.exists(DATA_FILE):
        print("Data file not found. Please run the original fake news script to download it first.")
        return
        
    print("Loading dataset...")
    df = pd.read_csv(DATA_FILE, index_col=0)
    
    # To save massive amounts of time for demonstration, we will only use a subset of 1000 items
    df = df.sample(n=1000, random_state=42)
    
    df['full_content'] = df['title'] + " " + df['text']
    y = df['label'].map({'REAL': 0, 'FAKE': 1}).values
    
    print("Extracting Neural BERT Embeddings... This is computationally intensive.")
    X_embeddings = extract_bert_embeddings(df['full_content'].tolist(), batch_size=16)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y, test_size=0.2, random_state=42)
    
    print("Training XGBoost Classifier over BERT features...")
    xgb_model = XGBClassifier(n_estimators=100, random_state=42, tree_method='hist', device='cuda' if torch.cuda.is_available() else 'cpu')
    xgb_model.fit(X_train, y_train)
    
    y_pred = xgb_model.predict(X_test)
    print(f"Accuracy with BERT Embeddings: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))
    
    # Save the specialized XGBoost model since the tokenizer/model are pulled natively from HuggingFace
    os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
    joblib.dump(xgb_model, os.path.join(BASE_DIR, 'models', 'fake_news_bert_xgboost.pkl'))
    print("Saved the BERT-powered XGBoost model to models/fake_news_bert_xgboost.pkl")

if __name__ == "__main__":
    main()
