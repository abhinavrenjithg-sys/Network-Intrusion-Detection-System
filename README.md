# 🛡️ Enterprise ML Security Portfolio (Full-Stack Edition)

<p align="center">
  <em>An interconnected, massive suite of ML, Deep Learning, and generative LLMs solving real-world cybersecurity problems.</em>
</p>

## Overview
This repository contains a suite of production-grade Machine Learning pipelines designed for the Cybersecurity and Financial sectors. 
As part of our **Phase 5 Architecture Upgrade**, all scripts have been decoupled into independent microservices, featuring **Kafka-streaming simulators**, **PyTorch Deep Learning**, and **LangChain / HuggingFace RAG Copilots**.

---

### 🚀 Flagship Systems

#### 1. Real-Time AI SOC & Deep Learning Pipeline
A high-throughput log ingestion engine (`src/soc_pipeline.py`) simulating enterprise Kafka queues.
- Utilizes **PyTorch LSTM Autoencoders** to evaluate long-tail memory sequences of IP packets to detect slow-burn APT anomalies.
- Utilizes **Scikit-Learn Isolation Forests** for instantaneous mathematical outlier flagging.
- Triggers are securely documented into a local `SQLite` state (simulating ElasticSearch) and projected onto a Role-Based **Streamlit Dashboard** (`app/soc_dashboard.py`).

#### 2. Threat Intel RAG Copilot (Local LLM)
An autonomous, context-aware cybersecurity assistant built natively using **LangChain** and **HuggingFace (`TinyLlama`)**.
- Uses `sentence-transformers` to chunk and embed simulated CVE and MITRE ATT&CK datasets into an offline **FAISS Vector Database**.
- Served via a **FastAPI** web socket connection and visualized in a sleek, NPM-free **React Single Page Application** (`app/rag_frontend.html`).
- The AI SOC autonomously invokes this RAG bot when anomalies are detected to dynamically generate incident response plans!

#### 3. Core Neural Pipelines (APIs & Dashboards)
- **Credit Card Fraud Identification**: A **Streamlit Live Dashboard** that pings a secondary FastAPI backend, plotting intercepted simulated financial transactions.
- **Network Intrusion Detection System (NIDS)**: A **Flask Web App** scanning `.csv` packet captures using a pre-compiled `XGBoost` classification matrix.
- **Malware Detection via PE Headers**: A **FastAPI** `/scan` endpoint that can accept raw `.exe` file uploads, dissect the Portable Executable dynamically, and infer malice using ExtraTrees isolation.
- **Fake News Detector (BERT)**: A **Streamlit Interface** hooked instantly into a HuggingFace Transformer (`distilbert-base-uncased`), passing extracted DL embeddings into XGBoost.

---

## 🏗️ Architecture & Tools

- **Backend / Streaming:** FastAPI, Flask, Python `multiprocessing.Queue` (Kafka simulation), SQLite (ElasticSearch simulation).
- **Front-End UIs:** React (CDN), Streamlit, Jinja2 (HTML/CSS).
- **Deep Learning / NLP:** HuggingFace `transformers`, PyTorch `nn.LSTM`, LangChain, FAISS.
- **Machine Learning Flow:** Scikit-Learn, XGBoost (`cuda`), Imbalanced-Learn (SMOTE).

---

## 📊 Resume Impact Highlights

* **Engineered an autonomous AI Security Operations Center (SOC) heavily simulating Kafka streaming workloads, isolating anomalous network threats automatically via a hybridized Scikit-Learn Isolation Forest and a PyTorch LSTM Autoencoder.**
* **Architected a complete Retrieval-Augmented Generation (RAG) backend using LangChain and offline FAISS indices to process synthetic CVE JSON payloads, answering critical Threat Intelligence queries completely offline via a local HuggingFace 1.1B LLM.**
* **Deployed full-stack ML, serving high-throughput inference across 5 separate domains using FastAPI, Flask, and interactive React/Streamlit admin dashboards engineered to circumvent massive Node.js/Java local dependencies.**

---

## ⚙️ Quick Start 

Ensure your environment is robustly configured for PyTorch and Transformers!

### Booting the AI SOC Pipeline:
```bash
python src/soc_pipeline.py
# (Leave this running to generate fake attacks and AI responses)
```
View the incidents via Streamlit:
```bash
streamlit run app/soc_dashboard.py
```

### Chatting with the Local Threat RAG Copilot:
Boot the LangChain FAISS Engine:
```bash
uvicorn api.rag_api:app --port 8003
```
Simply open `app/rag_frontend.html` in your browser!

### Running the Phase 4 Apps:
- Fraud Backend: `uvicorn api.fraud_api:app --port 8001`
- Fraud UI: `streamlit run app/fraud_dashboard.py`
- Malware Sandbox: `uvicorn api.malware_api:app --port 8002`

---
*Built with ❤️ in Python. If you find this useful, please leave a ⭐!*
