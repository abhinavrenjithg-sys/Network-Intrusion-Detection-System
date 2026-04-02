import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import random

st.set_page_config(page_title="ThreatWatch: Fraud Stream", layout="wide", page_icon="💳")

# Connect to the FastAPI Fraud Backend
API_URL = "http://localhost:8001/predict"

st.title("💳 Real-Time Credit Card Fraud Operations")
st.markdown("This dashboard tracks live network latency and pings our locally hosted **FastAPI** model to isolate simulated transactions passing through our banking architecture.")

# Initialize Session State Arrays for the dynamic charts
if 'history' not in st.session_state:
    st.session_state.history = {'Legitimate': 0, 'Fraudulent': 0}
if 'log' not in st.session_state:
    st.session_state.log = []

# Layout Setup
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Simulate Incoming Trx Stream")
    if st.button("▶ Start Pinging Inference Engine", type="primary"):
        # We will iterate 20 transactions
        st.write("Intercepting live traffic...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(20):
            # Generate a completely synthetic feature array matching the 28 PCAs
            # Inject a 10% chance of extreme anomaly (simulating possible fraud)
            base = np.random.normal(0, 1, 28).tolist()
            if random.random() < 0.15:
                base = np.random.normal(5, 5, 28).tolist() # Heavy outliers
                
            payload = {
                "Time": time.time(),
                "Amount": random.uniform(1.0, 5000.0)
            }
            # Append V1-V28
            for j in range(28):
                payload[f"V{j+1}"] = base[j]
                
            # Ping Backend
            try:
                res = requests.post(API_URL, json=payload, timeout=2).json()
                
                # Update State
                if res.get("is_fraud"):
                    st.session_state.history['Fraudulent'] += 1
                    status = "🔴 BLOCKED"
                else:
                    st.session_state.history['Legitimate'] += 1
                    status = "🟢 AUTHORIZED"
                    
                st.session_state.log.insert(0, {
                    "Amount": payload["Amount"], 
                    "Status": status, 
                    "Confidence (Fraud)": f"{res.get('fraud_confidence', 0)*100:.2f}%"
                })
                
            except requests.exceptions.ConnectionError:
                st.error("API SERVER OFFLINE. Please boot the Fraud API using `uvicorn api.fraud_api:app --port 8001`.")
                st.stop()
                
            progress_bar.progress(min(100, int((i+1)*5)))
            time.sleep(0.5) # Simulate slight streaming delay
            
        status_text.success("Scan chunk completed!")

with col2:
    st.subheader("Real-Time Tally")
    metrics_col1, metrics_col2 = st.columns(2)
    metrics_col1.metric("Legitimate Auth", st.session_state.history['Legitimate'])
    metrics_col2.metric("Blocked Frauds", st.session_state.history['Fraudulent'], delta_color="inverse")
    
    st.subheader("Transaction Terminal Log")
    if len(st.session_state.log) > 0:
        st.dataframe(pd.DataFrame(st.session_state.log).head(20), use_container_width=True)
        
    if st.button("Clear Cache"):
        st.session_state.history = {'Legitimate': 0, 'Fraudulent': 0}
        st.session_state.log = []
        st.rerun()
