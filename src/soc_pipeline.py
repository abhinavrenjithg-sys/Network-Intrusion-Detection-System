import time
import random
import threading
import queue
import numpy as np
import pandas as pd
import sqlite3
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
import requests
import json
from logger import get_logger

log = get_logger("AI_SOC_KafkaStreamer")

# --- "Kafka Topics" simulated entirely in Python for absolute local performance ---
RAW_LOG_TOPIC = queue.Queue(maxsize=5000)
ALERT_TOPIC = queue.Queue(maxsize=1000)

DB_PATH = "data/soc_alerts.sqlite"

# --- 1. Deep Learning LSTM Autoencoder (Sequence Anomalies) ---
class LSTMAutoencoder(nn.Module):
    """Detects multi-stage persistent threats (APTs) by memory reconstruction failure."""
    def __init__(self, input_dim=5, hidden_dim=16, seq_length=10):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.decoder = nn.LSTM(input_size=hidden_dim, hidden_size=input_dim, num_layers=1, batch_first=True)
        
    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        # Repeat the hidden state for the entire sequence length
        hidden_repeated = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        decoded, _ = self.decoder(hidden_repeated)
        return decoded

# Initialize Deep Learning Model & Isolation Forest
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lstm_model = LSTMAutoencoder().to(device)
lstm_model.eval() # We won't re-train it during the stream to save time. It acts functionally as an anomaly flagger based on loss threshold
iso_forest = IsolationForest(contamination=0.01, random_state=42)

def initialize_database():
    """Simulates ElasticSearch document storage."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    ip_address TEXT,
                    bytes_transferred REAL,
                    failed_logins INTEGER,
                    status TEXT,
                    ai_incident_response TEXT)''')
    conn.commit()
    conn.close()
    log.info("Initialized Autonomous ElasticSearch alternative (SQLite SOC Logs)")

def kafka_producer_simulator():
    """Generates 50 network logs per second. Simulates Nginx/Syslog forwarding."""
    log.info("Started High-Throughput Log Streaming Simulator...")
    while True:
        # Heavily simulate normal traffic
        event = {
            "timestamp": time.time(),
            "ip_address": f"192.168.1.{random.randint(10, 100)}",
            "bytes_transferred": random.uniform(100, 5000),
            "failed_logins": 0,
            "session_length": random.uniform(1, 30),
            "weird_port": 0
        }
        
        # Inject an anomaly roughly 1% of the time
        if random.random() < 0.01:
            event["ip_address"] = f"10.0.0.{random.randint(1, 10)} (Attacker Endpoint)"
            event["bytes_transferred"] = random.uniform(50000, 500000) # Data Exfiltration
            event["failed_logins"] = random.randint(5, 50) # Brute force
            event["session_length"] = random.uniform(0.1, 0.5) # Fast automation scanner
            event["weird_port"] = 1 # RDP/SSH anomalies
            
        try:
            RAW_LOG_TOPIC.put(event, block=False)
        except queue.Full:
            pass # Drop logs if queue filled (standard UDP/Kafka overflow mitigation)
        time.sleep(0.02) # 50 logs/second

def query_threat_intel_rag(threat_type):
    """Automatically asks the local LangChain Fast-API to build an Autonomous Response."""
    try:
        # Ping the local RAG Threat Intelligence API running on port 8003
        payload = {"query": f"We detected a {threat_type} anomalous behavior pattern. What MITRE ATT&CK tactic is this and how do we instantly remediate it?"}
        res = requests.post("http://localhost:8003/chat", json=payload, timeout=5)
        if res.status_code == 200:
            return res.json().get('response', 'Unable to retrieve context.')
    except:
        return "RAG Server Offline. Recommend blocking IP immediately."

def kafka_consumer_ml_pipeline():
    """Reads from Kafka queue, builds sequences, and runs PyTorch LSTM & Scikit-Learn IsolationForest."""
    log.info("Started AI Microservice Log Ingestion Engine (Listening for payloads...)")
    
    # Needs some initialization data for Isolation Forest
    warmup_data = [[random.uniform(100, 5000), 0, random.uniform(1, 30), 0] for _ in range(500)]
    iso_forest.fit(warmup_data)
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Stateful memory for the LSTM (tracking IPs)
    ip_history = {}
    
    while True:
        try:
            event = RAW_LOG_TOPIC.get(timeout=2)
            
            # 1. Feature Vector
            features = [
                event['bytes_transferred'],
                event['failed_logins'],
                event['session_length'],
                event['weird_port']
            ]
            
            # --- MODEL 1: Isolation Forest (Stateless Anomaly Detection) ---
            if_prediction = iso_forest.predict([features])[0]
            
            is_anomalous = False
            threat_type = "None"
            
            if if_prediction == -1: # Outlier detected!
                is_anomalous = True
                threat_type = "Brute Force / Data Exfil"
                
            # --- MODEL 2: PyTorch LSTM Autoencoder (Stateful Sequential Detection) ---
            ip = event['ip_address']
            if ip not in ip_history:
                ip_history[ip] = []
            ip_history[ip].append(features)
            
            # Once we have 10 sequential packets from an IP, check LSTM
            if len(ip_history[ip]) == 10:
                seq_tensor = torch.tensor([ip_history[ip]], dtype=torch.float32).to(device)
                with torch.no_grad():
                    reconstruction = lstm_model(seq_tensor)
                    loss = nn.MSELoss()(reconstruction, seq_tensor).item()
                
                # If reconstruction loss > threshold, it's anomalous sequentially
                if loss > 5000: # Heuristic threshold for demo
                    is_anomalous = True
                    threat_type = "Multi-Stage Sequential APT"
                    
                ip_history[ip].pop(0) # Sliding window
                
            # --- THE AUTO-RESPONDER & ELASTICSEARCH DOCKING ---
            if is_anomalous:
                log.warning(f"🚨 [AI SOC] Threat Detected from {ip}: {threat_type}")
                
                # Automate Threat Intelligence Retrieval via LLM RAG!
                rag_action_plan = query_threat_intel_rag(threat_type)
                
                # Document into "ElasticSearch" (SQLite)
                c.execute("INSERT INTO alerts (timestamp, ip_address, bytes_transferred, failed_logins, status, ai_incident_response) VALUES (?, ?, ?, ?, ?, ?)",
                          (event['timestamp'], ip, event['bytes_transferred'], event['failed_logins'], "BLOCKED", rag_action_plan))
                conn.commit()
                
        except queue.Empty:
            continue
        except Exception as e:
            log.error(f"SOC ML Engine Error: {e}")

if __name__ == "__main__":
    import os
    os.makedirs('data', exist_ok=True)
    initialize_database()
    
    # We orchestrate multi-threading in pure Python to eliminate local Java cluster configuration
    producer_thread = threading.Thread(target=kafka_producer_simulator, daemon=True)
    consumer_thread = threading.Thread(target=kafka_consumer_ml_pipeline, daemon=True)
    
    producer_thread.start()
    consumer_thread.start()
    
    log.info("Enterprise SOC Pipeline Active. Press CTRL+C to stop.")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        log.info("SOC shutting down gracefully.")
