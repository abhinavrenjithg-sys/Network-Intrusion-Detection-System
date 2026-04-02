"""
ThreatWatch AI — One-Click Launcher
Boots ALL services in parallel then opens the hub in your browser.
"""
import subprocess, sys, os, time, webbrowser, signal

BASE = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.join(BASE, "venv", "Scripts", "python.exe")

SERVICES = [
    {
        "name": "SOC Pipeline       (Kafka Streamer + LSTM)",
        "cmd":  [PYTHON, "src/soc_pipeline.py"],
        "delay": 0,
    },
    {
        "name": "Fraud API          (FastAPI :8001)",
        "cmd":  [PYTHON, "-m", "uvicorn", "api.fraud_api:app", "--port", "8001", "--host", "0.0.0.0"],
        "delay": 1,
    },
    {
        "name": "Malware API        (FastAPI :8002)",
        "cmd":  [PYTHON, "-m", "uvicorn", "api.malware_api:app", "--port", "8002", "--host", "0.0.0.0"],
        "delay": 1,
    },
    {
        "name": "Threat Intel API   (FastAPI :8003 + RAG LLM)",
        "cmd":  [PYTHON, "-m", "uvicorn", "api.rag_api:app", "--port", "8003", "--host", "0.0.0.0"],
        "delay": 2,
    },
    {
        "name": "NIDS Flask App     (:5000)",
        "cmd":  [PYTHON, "app/nids_flask_app.py"],
        "delay": 1,
    },
]

procs = []

def shutdown(sig=None, frame=None):
    print("\n\n[ThreatWatch] Shutting down all services...")
    for p in procs:
        try:
            p.terminate()
        except:
            pass
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)

print("=" * 60)
print("  🛡️  ThreatWatch AI — Enterprise Cybersecurity Suite")
print("=" * 60)
print()

for svc in SERVICES:
    time.sleep(svc["delay"])
    print(f"  ▶ Starting  {svc['name']}...")
    p = subprocess.Popen(
        svc["cmd"],
        cwd=BASE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    procs.append(p)

print()
print("  ⏳ Waiting for APIs to finish booting (8 seconds)...")
time.sleep(8)

print()
print("  🚀 Launching ThreatWatch Hub Dashboard...")
hub_proc = subprocess.Popen(
    [PYTHON, "-m", "streamlit", "run", "app/hub.py",
     "--server.port", "8501",
     "--server.headless", "false",
     "--browser.gatherUsageStats", "false"],
    cwd=BASE,
)
procs.append(hub_proc)

time.sleep(4)
webbrowser.open("http://localhost:8501")

print()
print("=" * 60)
print("  ✅ ALL SYSTEMS OPERATIONAL")
print()
print("  Hub Dashboard  →  http://localhost:8501")
print("  Fraud API      →  http://localhost:8001/docs")
print("  Malware API    →  http://localhost:8002/docs")
print("  RAG API        →  http://localhost:8003/docs")
print("  NIDS App       →  http://localhost:5000")
print()
print("  Press CTRL+C to shut down everything.")
print("=" * 60)

try:
    while True:
        time.sleep(1)
        # Restart any crashed service
        for i, (svc, p) in enumerate(zip(SERVICES, procs[:-1])):
            if p.poll() is not None:  # process died
                print(f"  ⚠️  {svc['name']} crashed. Restarting...")
                new_p = subprocess.Popen(
                    svc["cmd"], cwd=BASE,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                )
                procs[i] = new_p
except KeyboardInterrupt:
    shutdown()
