import os
import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import joblib
import numpy as np

# Streamlit config (must be first command)
st.set_page_config(page_title="Fake News Linguistic Detector", page_icon="📰", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fake_news_bert_xgboost.pkl")

st.title("📰 Real-Time Fake News Detector (BERT-Powered)")
st.markdown("This application uses an AI transformer (**HuggingFace DistilBERT**) chained to an **XGBoost Classifier** to capture complex sarcasm and linguistic nuances to classify political texts.")

@st.cache_resource
def load_bert_assets():
    st.info("Loading DistilBERT Embeddings (This takes a moment...)")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    bert = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
    bert.eval()
    return tokenizer, bert, device

@st.cache_resource
def load_xgb_model():
    return joblib.load(MODEL_PATH)

try:
    tokenizer, bert_model, device = load_bert_assets()
    xgb_model = load_xgb_model()
    st.success("✅ Deep Learning Pipeline Initialized!")
except Exception as e:
    st.error(f"Failed to boot Neural Network pipeline: {e}")
    st.stop()

# UI Inputs
news_input = st.text_area("Paste the News Article or Headline here:", height=300, placeholder="The President today announced that aliens have landed...")

if st.button("Predict Authenticity", type="primary"):
    if len(news_input.strip()) < 10:
        st.warning("Please enter a longer piece of text for reliable NLP analysis.")
    else:
        with st.spinner("Extracting hidden linguistic features using BERT..."):
            # 1. Tokenize the input string exactly as we did in training
            inputs = tokenizer([news_input], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            
            # 2. Extract [CLS] Embeddings
            with torch.no_grad():
                outputs = bert_model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
            # 3. Predict via XGBoost
            prediction = xgb_model.predict(cls_embedding)[0]
            confidence = xgb_model.predict_proba(cls_embedding)[0]
            
            # 0 = Real, 1 = Fake
            st.divider()
            if prediction == 1:
                st.error(f"🚨 **FAKE NEWS DETECTED** (Confidence: {confidence[1]*100:.2f}%)")
                st.markdown("🚨 *The Transformer model strongly suspects this article contains artificially generated, fabricated, or satirical content based on semantic anomalies.*")
            else:
                st.success(f"✅ **AUTHENTIC NEWS** (Confidence: {confidence[0]*100:.2f}%)")
                st.markdown("✅ *Linguistic analysis suggests this text matches standard journalistic structuring.*")
