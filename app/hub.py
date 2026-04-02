import streamlit as st
import os, sys, sqlite3, time, random, requests
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "soc_alerts.sqlite")

st.set_page_config(
    page_title="ThreatWatch AI — Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── GLOBAL DARK GLASSMORPHISM CSS ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* Dark background */
.stApp {
    background: linear-gradient(135deg, #020818 0%, #0a1628 40%, #0d1f3c 100%);
    min-height: 100vh;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(8, 20, 40, 0.95) !important;
    border-right: 1px solid rgba(99,102,241,0.2) !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* All containers / cards */
[data-testid="stVerticalBlock"] > div > [data-testid="stVerticalBlock"] {
    background: rgba(15,25,50,0.6);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(12px);
}

/* Metric cards */
[data-testid="metric-container"] {
    background: rgba(99,102,241,0.08) !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
[data-testid="metric-container"] label { color: #94a3b8 !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { 
    color: #6366f1 !important; font-weight: 700 !important; font-size: 2rem !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.5) !important;
}

/* Text inputs / Textareas */
.stTextArea textarea, .stTextInput input {
    background: rgba(15,25,50,0.8) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.2) !important;
}

/* Dataframes */
[data-testid="stDataFrame"] { 
    border-radius: 10px !important; 
    overflow: hidden !important; 
}

/* Alerts */
.stAlert { border-radius: 10px !important; }
.stSuccess { background: rgba(16,185,129,0.1) !important; border-color: #10b981 !important; }
.stError { background: rgba(239,68,68,0.1) !important; border-color: #ef4444 !important; }
.stWarning { background: rgba(245,158,11,0.1) !important; border-color: #f59e0b !important; }
.stInfo { background: rgba(99,102,241,0.1) !important; border-color: #6366f1 !important; }

/* Headers */
h1 { 
    background: linear-gradient(135deg, #818cf8, #6366f1, #4f46e5);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-weight: 800 !important; letter-spacing: -0.5px;
}
h2, h3 { color: #c7d2fe !important; font-weight: 700 !important; }
p, span, li { color: #94a3b8; }
label { color: #94a3b8 !important; }

/* Progress bar */
.stProgress > div > div { background: linear-gradient(90deg, #6366f1, #818cf8) !important; border-radius: 99px; }

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: rgba(15,25,50,0.8) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(15,25,50,0.5) !important;
    border: 2px dashed rgba(99,102,241,0.3) !important;
    border-radius: 12px !important;
}

/* Divider */
hr { border-color: rgba(99,102,241,0.15) !important; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR NAVIGATION ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 2rem 0;'>
        <div style='font-size:3rem;'>🛡️</div>
        <div style='font-size:1.1rem; font-weight:800; color:#818cf8; letter-spacing:1px;'>ThreatWatch AI</div>
        <div style='font-size:0.75rem; color:#64748b; margin-top:4px;'>Enterprise Security Suite</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "🏠  Command Center",
        "🌐  NIDS Scanner",
        "💳  Fraud Detection",
        "🦠  Malware Sandbox",
        "🤖  Threat Intel AI",
        "🛡️  Live SOC Monitor",
        "📰  Fake News Detector",
    ], label_visibility="collapsed")

    st.divider()
    
    # Service health panel
    st.markdown("<div style='font-size:0.8rem; font-weight:600; color:#64748b; letter-spacing:1px; text-transform:uppercase; margin-bottom:8px;'>Service Status</div>", unsafe_allow_html=True)
    
    def check_service(url, name):
        try:
            r = requests.get(url, timeout=1)
            st.markdown(f"<span style='color:#10b981;'>●</span> <span style='font-size:0.85rem; color:#94a3b8;'>{name}</span>", unsafe_allow_html=True)
        except:
            st.markdown(f"<span style='color:#ef4444;'>●</span> <span style='font-size:0.85rem; color:#94a3b8;'>{name}</span>", unsafe_allow_html=True)

    check_service("http://localhost:8001/docs", "Fraud API :8001")
    check_service("http://localhost:8002/docs", "Malware API :8002")
    check_service("http://localhost:8003/docs", "RAG API :8003")
    check_service("http://localhost:5000", "NIDS Flask :5000")
    
    soc_running = os.path.exists(DB_PATH)
    color = "#10b981" if soc_running else "#ef4444"
    st.markdown(f"<span style='color:{color};'>●</span> <span style='font-size:0.85rem; color:#94a3b8;'>SOC Pipeline</span>", unsafe_allow_html=True)

# ── PAGES ───────────────────────────────────────────────────────────────────

# ── HOME ────────────────────────────────────────────────────────────────────
if page == "🏠  Command Center":
    st.markdown("# 🛡️ ThreatWatch AI Command Center")
    st.markdown("<p style='color:#64748b; font-size:1.1rem; margin-bottom:2rem;'>Enterprise-grade machine learning cybersecurity suite — powered by XGBoost, PyTorch LSTM, DistilBERT, and LangChain RAG.</p>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    
    # Pull live DB stats
    fraud_total, nids_total, soc_alerts = 0, 0, 0
    try:
        conn = sqlite3.connect(DB_PATH)
        soc_alerts = pd.read_sql_query("SELECT COUNT(*) as c FROM alerts", conn)['c'][0]
        conn.close()
    except: pass

    col1.metric("🚨 SOC Alerts Fired", soc_alerts, "Live")
    col2.metric("🌐 NIDS Models", "2", "XGBoost + RF")
    col3.metric("💳 Fraud Models", "3", "LR + RF + XGB")
    col4.metric("🦠 Malware Features", "15", "ExtraTrees")

    st.divider()
    
    st.markdown("### 📦 System Modules")
    
    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        st.markdown("""
        <div style='background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.25); border-radius:14px; padding:1.5rem;'>
            <div style='font-size:2rem;'>🌐</div>
            <div style='font-weight:700; color:#c7d2fe; font-size:1rem; margin:8px 0 4px;'>NIDS Network Scanner</div>
            <div style='font-size:0.82rem; color:#64748b;'>Upload .csv network logs. XGBoost classifies each packet as safe or intrusion in milliseconds.</div>
            <div style='margin-top:10px;'><span style='background:rgba(16,185,129,0.15); color:#10b981; padding:3px 10px; border-radius:99px; font-size:0.75rem; font-weight:600;'>Flask :5000</span></div>
        </div>
        """, unsafe_allow_html=True)
    with r1c2:
        st.markdown("""
        <div style='background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.25); border-radius:14px; padding:1.5rem;'>
            <div style='font-size:2rem;'>💳</div>
            <div style='font-weight:700; color:#c7d2fe; font-size:1rem; margin:8px 0 4px;'>Fraud Detection Stream</div>
            <div style='font-size:0.82rem; color:#64748b;'>Live credit card transaction simulator. SMOTE-balanced XGBoost flags fraudulent payments instantly.</div>
            <div style='margin-top:10px;'><span style='background:rgba(99,102,241,0.2); color:#818cf8; padding:3px 10px; border-radius:99px; font-size:0.75rem; font-weight:600;'>FastAPI :8001</span></div>
        </div>
        """, unsafe_allow_html=True)
    with r1c3:
        st.markdown("""
        <div style='background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.25); border-radius:14px; padding:1.5rem;'>
            <div style='font-size:2rem;'>🦠</div>
            <div style='font-weight:700; color:#c7d2fe; font-size:1rem; margin:8px 0 4px;'>Malware PE Sandbox</div>
            <div style='font-size:0.82rem; color:#64748b;'>Upload .exe files. Dynamic PE header dissection + ExtraTrees feature isolation + XGBoost inference.</div>
            <div style='margin-top:10px;'><span style='background:rgba(239,68,68,0.15); color:#ef4444; padding:3px 10px; border-radius:99px; font-size:0.75rem; font-weight:600;'>FastAPI :8002</span></div>
        </div>
        """, unsafe_allow_html=True)

    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        st.markdown("""
        <div style='background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.25); border-radius:14px; padding:1.5rem;'>
            <div style='font-size:2rem;'>🤖</div>
            <div style='font-weight:700; color:#c7d2fe; font-size:1rem; margin:8px 0 4px;'>Threat Intel RAG Bot</div>
            <div style='font-size:0.82rem; color:#64748b;'>LangChain + FAISS + HuggingFace local LLM. Ask anything about CVEs and MITRE ATT&CK tactics.</div>
            <div style='margin-top:10px;'><span style='background:rgba(245,158,11,0.15); color:#f59e0b; padding:3px 10px; border-radius:99px; font-size:0.75rem; font-weight:600;'>FastAPI :8003</span></div>
        </div>
        """, unsafe_allow_html=True)
    with r2c2:
        st.markdown("""
        <div style='background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.25); border-radius:14px; padding:1.5rem;'>
            <div style='font-size:2rem;'>🛡️</div>
            <div style='font-weight:700; color:#c7d2fe; font-size:1rem; margin:8px 0 4px;'>AI SOC Live Monitor</div>
            <div style='font-size:0.82rem; color:#64748b;'>Real-time Kafka stream. IsolationForest + LSTM Autoencoder. Auto-generates AI incident response plans.</div>
            <div style='margin-top:10px;'><span style='background:rgba(16,185,129,0.15); color:#10b981; padding:3px 10px; border-radius:99px; font-size:0.75rem; font-weight:600;'>Background Process</span></div>
        </div>
        """, unsafe_allow_html=True)
    with r2c3:
        st.markdown("""
        <div style='background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.25); border-radius:14px; padding:1.5rem;'>
            <div style='font-size:2rem;'>📰</div>
            <div style='font-weight:700; color:#c7d2fe; font-size:1rem; margin:8px 0 4px;'>Fake News Detector</div>
            <div style='font-size:0.82rem; color:#64748b;'>DistilBERT 768-dim embeddings + XGBoost. Deep semantic analysis to classify political text.</div>
            <div style='margin-top:10px;'><span style='background:rgba(99,102,241,0.2); color:#818cf8; padding:3px 10px; border-radius:99px; font-size:0.75rem; font-weight:600;'>Transformer NLP</span></div>
        </div>
        """, unsafe_allow_html=True)

# ── FRAUD DETECTION ──────────────────────────────────────────────────────────
elif page == "💳  Fraud Detection":
    st.markdown("# 💳 Real-Time Fraud Detection")
    st.markdown("<p style='color:#64748b;'>Simulates live banking transactions and pings the XGBoost inference engine to block fraud instantaneously.</p>", unsafe_allow_html=True)

    API_URL = "http://localhost:8001/predict"

    if 'fraud_history' not in st.session_state:
        st.session_state.fraud_history = {'Legitimate': 0, 'Fraudulent': 0}
    if 'fraud_log' not in st.session_state:
        st.session_state.fraud_log = []

    col1, col2, col3 = st.columns([1, 1, 1])
    col1.metric("✅ Authorized", st.session_state.fraud_history['Legitimate'])
    col2.metric("🚨 Blocked", st.session_state.fraud_history['Fraudulent'])
    total = st.session_state.fraud_history['Legitimate'] + st.session_state.fraud_history['Fraudulent']
    fraud_rate = (st.session_state.fraud_history['Fraudulent'] / total * 100) if total > 0 else 0
    col3.metric("📊 Fraud Rate", f"{fraud_rate:.1f}%")

    st.divider()

    btn_col1, btn_col2 = st.columns([2, 1])
    with btn_col1:
        n_txns = st.slider("Transactions to simulate", 10, 100, 20, 5)
    with btn_col2:
        st.write("")
        st.write("")
        run_sim = st.button("▶ Run Simulation", type="primary", use_container_width=True)

    if run_sim:
        progress = st.progress(0, text="Intercepting transaction stream...")
        status_placeholder = st.empty()

        for i in range(n_txns):
            base = np.random.normal(0, 1, 28).tolist()
            if random.random() < 0.15:
                base = np.random.normal(5, 5, 28).tolist()
            payload = {"Time": time.time(), "Amount": random.uniform(1.0, 5000.0)}
            for j in range(28):
                payload[f"V{j+1}"] = base[j]

            try:
                res = requests.post(API_URL, json=payload, timeout=2).json()
                is_fraud = res.get("is_fraud", False)
                conf = res.get("fraud_confidence", 0)
                if is_fraud:
                    st.session_state.fraud_history['Fraudulent'] += 1
                    st.session_state.fraud_log.insert(0, {"#": i+1, "Amount": f"${payload['Amount']:.2f}", "Status": "🔴 BLOCKED", "Confidence": f"{conf*100:.1f}%"})
                else:
                    st.session_state.fraud_history['Legitimate'] += 1
                    st.session_state.fraud_log.insert(0, {"#": i+1, "Amount": f"${payload['Amount']:.2f}", "Status": "🟢 AUTHORIZED", "Confidence": f"{(1-conf)*100:.1f}%"})
            except requests.exceptions.ConnectionError:
                st.error("⚠️ Fraud API offline. Make sure it's running: `uvicorn api.fraud_api:app --port 8001`")
                break

            progress.progress((i + 1) / n_txns, text=f"Processing transaction {i+1}/{n_txns}...")
            time.sleep(0.3)

        progress.progress(1.0, text="✅ Simulation complete!")

    if st.session_state.fraud_log:
        st.divider()
        st.markdown("### 📋 Transaction Log")
        df_log = pd.DataFrame(st.session_state.fraud_log[:50])
        st.dataframe(df_log, use_container_width=True, hide_index=True)

    if st.button("🗑️ Clear History"):
        st.session_state.fraud_history = {'Legitimate': 0, 'Fraudulent': 0}
        st.session_state.fraud_log = []
        st.rerun()

# ── THREAT INTEL RAG ─────────────────────────────────────────────────────────
elif page == "🤖  Threat Intel AI":
    st.markdown("# 🤖 Threat Intelligence Copilot")
    st.markdown("<p style='color:#64748b;'>Powered by HuggingFace distilgpt2 + FAISS local vector DB. Ask anything about CVEs and MITRE ATT&CK — no internet required.</p>", unsafe_allow_html=True)

    API_URL = "http://localhost:8003/chat"

    if 'rag_chat' not in st.session_state:
        st.session_state.rag_chat = [
            {"role": "assistant", "content": "👋 Hello! I am your AI Cyber Threat Analyst. I have deep knowledge of CVEs, MITRE ATT&CK tactics, and security remediation. What can I help you investigate today?"}
        ]

    # Chat history display
    for msg in st.session_state.rag_chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask about CVEs, MITRE tactics, remediations...")
    
    # Quick question chips
    cols = st.columns(4)
    quick_qs = ["How do I mitigate Log4Shell?", "Explain MITRE T1566", "What is CVE-2023-34362?", "How to stop brute force?"]
    clicked_q = None
    for i, q in enumerate(quick_qs):
        with cols[i]:
            if st.button(q, use_container_width=True):
                clicked_q = q

    active_query = query or clicked_q

    if active_query:
        st.session_state.rag_chat.append({"role": "user", "content": active_query})
        with st.chat_message("user"):
            st.markdown(active_query)
        with st.chat_message("assistant"):
            with st.spinner("Searching FAISS knowledge base..."):
                try:
                    res = requests.post(API_URL, json={"query": active_query}, timeout=30).json()
                    answer = res.get("response", res.get("error", "No response"))
                    source = res.get("sources", "")
                    full_answer = f"{answer}\n\n*📚 {source}*" if source else answer
                    st.markdown(full_answer)
                    st.session_state.rag_chat.append({"role": "assistant", "content": full_answer})
                except requests.exceptions.ConnectionError:
                    err = "⚠️ RAG API offline. Run: `uvicorn api.rag_api:app --port 8003`"
                    st.error(err)
                    st.session_state.rag_chat.append({"role": "assistant", "content": err})

    if len(st.session_state.rag_chat) > 2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.rag_chat = [st.session_state.rag_chat[0]]
            st.rerun()

# ── SOC MONITOR ───────────────────────────────────────────────────────────────
elif page == "🛡️  Live SOC Monitor":
    st.markdown("# 🛡️ AI Security Operations Center")
    st.markdown("<p style='color:#64748b;'>Real-time anomaly detection via IsolationForest + PyTorch LSTM Autoencoder. Autonomous AI-generated incident responses.</p>", unsafe_allow_html=True)

    role = st.sidebar.selectbox("🔐 Access Level", ["Level 1 Analyst", "Level 3 Admin"])

    if st.button("🔄 Refresh Live Data", type="primary"):
        st.rerun()

    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM alerts ORDER BY id DESC LIMIT 100", conn)
        conn.close()

        if df.empty:
            st.info("🟡 SOC pipeline active. Waiting for first anomaly detection...")
            st.markdown("<p style='color:#64748b; font-size:0.9rem;'>Tip: The Isolation Forest catches ~1% of simulated traffic as anomalous. This may take a few minutes to first trigger.</p>", unsafe_allow_html=True)
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🚨 Total Alerts", len(df))
            col2.metric("🌐 Unique Attacker IPs", df['ip_address'].nunique())
            col3.metric("📦 Data Intercepted", f"{df['bytes_transferred'].sum()/1e6:.1f} MB")
            col4.metric("🔑 Failed Logins Blocked", int(df['failed_logins'].sum()))

            st.divider()

            if role == "Level 3 Admin":
                st.error("🔐 ADMIN MODE — Full incident data visible")
                display_df = df[['id', 'ip_address', 'bytes_transferred', 'failed_logins', 'status']].copy()
            else:
                st.success("👤 Analyst Mode — Restricted view")
                display_df = df[['ip_address', 'bytes_transferred', 'status']].copy()

            display_df.columns = [c.replace('_', ' ').title() for c in display_df.columns]
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("### 🤖 Latest AI Incident Response")
            latest = df.iloc[0]
            st.error(f"**Flagged IP:** `{latest['ip_address']}`")
            ai_resp = latest.get('ai_incident_response', 'No response generated.')
            if ai_resp:
                st.markdown(f"> {ai_resp}")

    except sqlite3.OperationalError:
        st.warning("⚠️ Database not found. Start the SOC pipeline: `python src/soc_pipeline.py`")
    except Exception as e:
        st.error(f"DB Error: {e}")

# ── MALWARE SANDBOX ────────────────────────────────────────────────────────────
elif page == "🦠  Malware Sandbox":
    st.markdown("# 🦠 Malware PE Header Sandbox")
    st.markdown("<p style='color:#64748b;'>Upload any Windows executable. The scanner dynamically rips PE headers using `pefile` and runs XGBoost inference on the extracted features.</p>", unsafe_allow_html=True)

    API_URL = "http://localhost:8002/scan"

    uploaded = st.file_uploader("Drop a Windows executable (.exe / .dll)", type=["exe", "dll"])

    if uploaded:
        with st.spinner("🔬 Extracting PE headers and running XGBoost..."):
            try:
                res = requests.post(API_URL, files={"file": (uploaded.name, uploaded.getvalue(), "application/octet-stream")}, timeout=15).json()
                
                st.divider()
                status = res.get("status", "UNKNOWN")
                conf = res.get("malware_confidence", 0)
                
                if status == "MALWARE DETECTED":
                    st.error(f"## 🚨 MALWARE DETECTED")
                    st.error(f"**File:** `{res.get('filename')}` | **Confidence:** {conf*100:.1f}%")
                    st.warning("⚠️ Recommendation: Quarantine this file immediately. Do NOT execute.")
                else:
                    st.success(f"## ✅ FILE APPEARS SAFE")
                    st.success(f"**File:** `{res.get('filename')}` | **Clean Confidence:** {(1-conf)*100:.1f}%")

                feats = res.get("extracted_features", {})
                if feats:
                    st.divider()
                    st.markdown("### 🔍 Extracted PE Features")
                    feat_df = pd.DataFrame(list(feats.items()), columns=["Feature", "Value"])
                    st.dataframe(feat_df, use_container_width=True, hide_index=True)
            except requests.exceptions.ConnectionError:
                st.error("⚠️ Malware API offline. Run: `uvicorn api.malware_api:app --port 8002`")

# ── NIDS SCANNER ───────────────────────────────────────────────────────────────
elif page == "🌐  NIDS Scanner":
    st.markdown("# 🌐 Network Intrusion Detection")
    st.markdown("<p style='color:#64748b;'>Upload a CSV file of network packet data. The XGBoost classifier scans each packet and flags intrusions. CSV must have 42 numerical columns matching the KDD Cup 99 format.</p>", unsafe_allow_html=True)

    FLASK_URL = "http://localhost:5000"

    st.info(f"🔗 **Note:** The NIDS Flask app runs separately at [{FLASK_URL}]({FLASK_URL}). You can also use it directly below via the embedded scanner.")

    try:
        check = requests.get(FLASK_URL, timeout=2)
        st.success("✅ NIDS Flask server is online!")
    except:
        st.error("⚠️ NIDS server offline. The launcher should have started it. Check your terminal.")

    uploaded_csv = st.file_uploader("Upload KDD Cup 99 format CSV (42 numerical columns, no headers)", type=["csv"])

    if uploaded_csv:
        import joblib
        import pandas as pd
        try:
            model_path = os.path.join(BASE_DIR, "models", "nids_xgboost_model.pkl")
            scaler_path = os.path.join(BASE_DIR, "models", "nids_scaler.pkl")
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            df = pd.read_csv(uploaded_csv, header=None)
            scaled = scaler.transform(df.values)
            preds = model.predict(scaled)
            probs = model.predict_proba(scaled)[:, 1]
            
            n_attacks = int(preds.sum())
            n_safe = len(preds) - n_attacks

            col1, col2, col3 = st.columns(3)
            col1.metric("📦 Total Packets", len(preds))
            col2.metric("✅ Legitimate", n_safe)
            col3.metric("🚨 Intrusions Detected", n_attacks)

            results_df = pd.DataFrame({
                "Packet #": range(1, len(preds)+1),
                "Classification": ["🔴 ATTACK" if p==1 else "🟢 SAFE" for p in preds],
                "Confidence": [f"{(prob if pred==1 else 1-prob)*100:.1f}%" for pred, prob in zip(preds, probs)]
            })
            st.dataframe(results_df.head(50), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Scan failed: {e}")

# ── FAKE NEWS ──────────────────────────────────────────────────────────────────
elif page == "📰  Fake News Detector":
    st.markdown("# 📰 Fake News Linguistic Detector")
    st.markdown("<p style='color:#64748b;'>Powered by HuggingFace DistilBERT (768-dim embeddings) chained into an XGBoost classifier. Detects semantic anomalies in political text.</p>", unsafe_allow_html=True)

    MODEL_PATH = os.path.join(BASE_DIR, "models", "fake_news_bert_xgboost.pkl")

    if not os.path.exists(MODEL_PATH):
        st.warning("⚠️ BERT XGBoost model not found. Run `python src/run_fake_news_bert.py` to train it first.")
        st.stop()

    @st.cache_resource
    def load_all():
        import torch
        from transformers import DistilBertTokenizer, DistilBertModel
        import joblib
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        bert = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
        bert.eval()
        xgb_model = joblib.load(MODEL_PATH)
        return tokenizer, bert, device, xgb_model

    try:
        tokenizer, bert_model, device, xgb_model = load_all()
        st.success("✅ DistilBERT + XGBoost pipeline loaded")
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

    news_input = st.text_area("Paste news article or headline:", height=200, placeholder="Breaking: Scientists announce...")

    if st.button("🔍 Analyze Text", type="primary"):
        if len(news_input.strip()) < 10:
            st.warning("Please enter at least a few sentences for reliable analysis.")
        else:
            with st.spinner("Running deep semantic analysis with DistilBERT..."):
                import torch
                inputs = tokenizer([news_input], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = bert_model(**inputs)
                    cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                prediction = xgb_model.predict(cls_emb)[0]
                confidence = xgb_model.predict_proba(cls_emb)[0]

            st.divider()
            if prediction == 1:
                st.error(f"## 🚨 HIGH PROBABILITY: FAKE NEWS")
                st.error(f"**Fake Confidence: {confidence[1]*100:.1f}%** | Real Probability: {confidence[0]*100:.1f}%")
                st.markdown("*The transformer model detected semantic patterns inconsistent with authentic journalism.*")
            else:
                st.success(f"## ✅ LIKELY AUTHENTIC NEWS")
                st.success(f"**Real Confidence: {confidence[0]*100:.1f}%** | Fake Probability: {confidence[1]*100:.1f}%")
                st.markdown("*Linguistic structure matches authentic journalistic patterns.*")
