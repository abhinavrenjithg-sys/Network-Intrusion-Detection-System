import os
import time
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, precision_recall_curve
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib

from logger import get_logger

log = get_logger("FraudDetectorPipeline")

DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
DATA_FILE = "data/creditcard.csv"

def download_data():
    """Download the dataset defensively with nested error handling."""
    os.makedirs('data', exist_ok=True)
    if os.path.exists(DATA_FILE):
        log.info(f"Dataset already exists at {DATA_FILE}. Skipping download.")
        return
    
    log.info("Initiating dataset download (~150MB) - please wait...")
    try:
        start_time = time.time()
        response = requests.get(DATA_URL, timeout=120)
        response.raise_for_status()
        
        with open(DATA_FILE, 'wb') as f:
            f.write(response.content)
            
        log.info(f"Download complete in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        log.error(f"Critical error downloading data from {DATA_URL}: {e}")
        raise

def plot_pr_curve(y_test, y_prob, model_name):
    """Plots Precision-Recall Curve which is better for extremely imbalanced datasets."""
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label=model_name)
    plt.xlabel('Recall (Fraud Detection Rate)')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve: {model_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/fraud_pr_curve_{model_name.replace(' ', '_').lower()}.png")
    plt.close()

def evaluate_and_plot(y_test, y_pred, y_prob, model_name):
    """Calculates metrics heavily focused on Recall and generates confusion matrix."""
    os.makedirs('results', exist_ok=True)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred) # The critical metric
    f1 = f1_score(y_test, y_pred)
    
    log.info(f"--- Evaluation: {model_name} ---")
    log.info(f"Accuracy:  {acc:.4f} (Deceptive on imbalanced data)")
    log.info(f"Precision: {prec:.4f}")
    log.info(f"Recall:    {rec:.4f} <-- (Critical metric: % of fraud caught)")
    log.info(f"F1-Score:  {f1:.4f}")
    
    if y_prob is not None:
        roc_auc = roc_auc_score(y_test, y_prob)
        log.info(f"ROC-AUC:   {roc_auc:.4f}")
        plot_pr_curve(y_test, y_prob, model_name)
        
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=['Legitimate', 'Fraud'], yticklabels=['Legitimate', 'Fraud'])
    plt.title(f'Confusion Matrix: {model_name}\n(Focus: Lower Left corner minimizing)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"results/fraud_cm_{model_name.replace(' ', '_').lower()}.png")
    plt.close()


def main():
    log.info("Starting Credit Card Fraud Detection Pipeline...")
    
    try:
        download_data()
    except Exception:
        return
        
    log.info("Loading huge dataset into Pandas...")
    df = pd.read_csv(DATA_FILE)
    log.info(f"Dataset loaded. Shape: {df.shape}")
    
    # Analyze the massive imbalance
    fraud_count = len(df[df['Class'] == 1])
    legit_count = len(df[df['Class'] == 0])
    log.info(f"Data Distribution -> Legitimate: {legit_count}, Fraud: {fraud_count} ({(fraud_count/len(df))*100:.3f}%)")
    
    # Features & Targets
    # Time and Amount are not PCA reduced in the dataset, rest are V1-V28
    y = df['Class'].values
    X_df = df.drop(columns=['Class'])
    
    log.info("Scaling 'Time' and 'Amount' features...")
    scaler = StandardScaler()
    X_df['Amount'] = scaler.fit_transform(X_df['Amount'].values.reshape(-1, 1))
    X_df['Time'] = scaler.fit_transform(X_df['Time'].values.reshape(-1, 1))
    X = X_df.values
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/fraud_scaler.pkl")
    log.info("Saved Scaler to models/fraud_scaler.pkl")
    
    # Train test split (stratified is absolutely critical here)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Handle Severe Class Imbalance using SMOTE
    log.info("Applying SMOTE to fundamentally resolve the massive class imbalance during training.")
    smote = SMOTE(random_state=42)
    start_smote = time.time()
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    log.info(f"SMOTE completed in {time.time() - start_smote:.1f}s. Pre-SMOTE Fraud: {sum(y_train==1)}, Post-SMOTE Fraud: {sum(y_train_res==1)}")
    
    # Train Models
    # We prioritize recall in logistic regression by tweaking class_weights natively or using SMOTE. SMOTE handles it here.
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10), # Depth limited to prevent overfitting on SMOTE data
        "XGBoost (GPU)": XGBClassifier(n_estimators=100, random_state=42, tree_method='hist', device='cuda', eval_metric='auc')
    }
    
    for name, model in models.items():
        log.info(f"\nTraining Model: {name}")
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
        
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        evaluate_and_plot(y_test, y_pred, y_prob, name)
        
        if name == "XGBoost (GPU)":
            joblib.dump(model, "models/fraud_xgboost_model.pkl")
            log.info("Saved XGBoost model to models/fraud_xgboost_model.pkl")

    log.info("Fraud Detection Pipeline completed successfully.")

if __name__ == "__main__":
    main()
