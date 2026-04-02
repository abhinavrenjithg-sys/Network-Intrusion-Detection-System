import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_kddcup99
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings("ignore")

# Create a directory for outputs
os.makedirs("results", exist_ok=True)

def load_and_preprocess_data():
    print("1. Loading KDD Cup 99 dataset (10% subset)...")
    # Using the 10% subset for faster demonstration. fetch_kddcup99 handles downloading.
    # returns X and y. Data is numpy arrays.
    data = fetch_kddcup99(percent10=True)
    X_raw, y_raw = data.data, data.target
    
    # Convert byte strings to regular strings
    y_str = np.array([lbl.decode() if isinstance(lbl, bytes) else lbl for lbl in y_raw])
    
    # Create DataFrame for easier EDA
    # KDD Cup 99 has 41 features
    feature_names = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", 
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", 
        "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", 
        "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", 
        "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", 
        "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", 
        "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", 
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", 
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", 
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
    ]
    
    df = pd.DataFrame(X_raw, columns=feature_names)
    df['target'] = y_str
    
    print(f"Dataset Shape: {df.shape}")
    
    return df, feature_names

def perform_eda(df):
    print("\n2. Performing Exploratory Data Analysis (EDA)...")
    
    # 2.1 Distribution of labels (Top 10)
    plt.figure(figsize=(10, 6))
    top_10_labels = df['target'].value_counts().head(10)
    sns.barplot(x=top_10_labels.values, y=top_10_labels.index, palette='viridis')
    plt.title('Top 10 Attack Types (including normal)')
    plt.xlabel('Count')
    plt.ylabel('Label')
    plt.tight_layout()
    plt.savefig('results/top_10_labels.png')
    plt.close()
    
    # Convert to binary classification: 0 = normal, 1 = attack
    df['is_attack'] = df['target'].apply(lambda x: 0 if x == 'normal.' else 1)
    
    # 2.2 Plot binary distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='is_attack', palette='Set2')
    plt.title('Normal vs Attack Distribution')
    plt.xticks(ticks=[0, 1], labels=['Normal (0)', 'Attack (1)'])
    plt.savefig('results/binary_distribution.png')
    plt.close()
    
    print("EDA Visualizations saved in 'results/' directory.")
    return df

def preprocess_features(df, feature_names):
    print("\n3. Preprocessing Data...")
    
    # We drop the string 'target' and keep 'is_attack' as our label
    y = df['is_attack'].values
    X_df = df[feature_names].copy()
    
    # Identify categorical columns (in kdd99 they are bytes originally, pandas might treat them as object)
    # The categorical features in KDD are usually protocol_type, service, and flag
    categorical_cols = ['protocol_type', 'service', 'flag']
    
    for col in categorical_cols:
        if X_df[col].dtype == object or str(X_df[col].dtype).startswith('bytes'):
            X_df[col] = X_df[col].apply(lambda x: x.decode() if isinstance(x, bytes) else x)
            # Label encode
            le = LabelEncoder()
            X_df[col] = le.fit_transform(X_df[col].astype(str))
            
    # For remaining bytes columns (if any), cast to numeric
    for col in X_df.columns:
        if str(X_df[col].dtype) == 'object':
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce').fillna(0)
            
    X = X_df.values
    
    print(f"Features after encoding shape: {X.shape}")
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Normalize features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/nids_scaler.pkl")
    print("Scaler saved to models/nids_scaler.pkl")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X_df.columns

def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names):
    print("\n4. Handling Class Imbalance with SMOTE...")
    # Smote to oversample the minority class
    smote = SMOTE(random_state=42)
    # WARNING: This might take a few seconds on large datasets
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"Distribution before SMOTE: Normal: {sum(y_train==0)}, Attack: {sum(y_train==1)}")
    print(f"Distribution after SMOTE: Normal: {sum(y_train_sm==0)}, Attack: {sum(y_train_sm==1)}")
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost (GPU)": XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', tree_method='hist', device='cuda')
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n5. Training {name}...")
        model.fit(X_train_sm, y_train_sm)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"--- {name} Results ---")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        results[name] = model
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Attack'], 
                    yticklabels=['Normal', 'Attack'])
        plt.title(f'Confusion Matrix: {name}')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'results/conf_matrix_{name.replace(" ", "_").lower()}.png')
        plt.close()
        
        # Feature Importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], palette='magma')
            plt.title(f'Top 10 Feature Importances: {name}')
            plt.xlabel('Relative Importance')
            plt.tight_layout()
            plt.savefig(f'results/feature_importance_{name.replace(" ", "_").lower()}.png')
            plt.close()
            
        if name == "XGBoost (GPU)":
            joblib.dump(model, "models/nids_xgboost_model.pkl")
            print("Model saved to models/nids_xgboost_model.pkl")

if __name__ == "__main__":
    df, feature_names = load_and_preprocess_data()
    df = perform_eda(df)
    X_train, X_test, y_train, y_test, column_names = preprocess_features(df, feature_names)
    train_and_evaluate(X_train, X_test, y_train, y_test, column_names)
    print("\nPipeline execution complete! Check 'results' folder for plots.")
