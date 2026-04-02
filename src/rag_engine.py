import os
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from logger import get_logger

log = get_logger("CyberRAG-Engine")

THREAT_DB_PATH = "data/threat_intel.txt"

def build_synthetic_cyber_knowledge():
    os.makedirs('data', exist_ok=True)
    if os.path.exists(THREAT_DB_PATH): return
    
    intel = """CVE-2023-34362: MOVEit Transfer SQL Injection vulnerability. A critical vulnerability in Progress MOVEit Transfer allows unauthenticated remote attackers to gain unauthorized access to the database. CVSS: 9.8 (Critical). Remediation: Immediately apply the vendor patches, block external access to HTTP/HTTPS ports used by the transfer protocol if unpatched, and rotate service account credentials.

CVE-2021-44228: Log4Shell. An unauthenticated remote code execution vulnerability found in Apache Log4j 2. Attackers can execute arbitrary code by supplying a maliciously formatted JNDI string. CVSS: 10.0 (Critical). Remediation: Upgrade log4j2 to version 2.15.0 or later, implement WAF rules to block "${jndi:" strings.

MITRE ATT&CK T1078: Valid Accounts. Cyber adversaries frequently use stolen credentials or active valid accounts to bypass access controls and evade generic signature-based detection. Remediation: Mandate Multi-Factor Authentication (MFA), actively monitor and automatically block Impossible Travel logins, and conduct routine auditing of dormant Active Directory accounts.

MITRE ATT&CK T1566: Phishing. Adversaries send deceptive emails containing malicious attachments or links to invoke User Execution. Spearphishing attachments often contain embedded macros or PE executables. Remediation: Implement secure email gateways (SEG), detonate suspected attachments in a sandbox, and invoke an AI-driven NLP classification system to quarantine deceptive conversational hooks."""
    
    with open(THREAT_DB_PATH, 'w', encoding='utf-8') as f:
        f.write(intel)

class ManualRAGChain:
    def __init__(self):
        build_synthetic_cyber_knowledge()
        
        log.info("Booting Pure RAG Strategy (Native HF/FAISS)...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        with open(THREAT_DB_PATH, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        self.docs = raw_text.split('\n\n')
        
        # Build FAISS natively
        embeddings = self.embedding_model.encode(self.docs)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        
        # Build LLM
        model_id = "distilgpt2"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # distilgpt2 is purely a float32 model natively, prevents CPU crashes
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.llm = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True).to(device)
        self.pipe = pipeline(
            "text-generation", 
            model=self.llm, 
            tokenizer=self.tokenizer, 
            max_new_tokens=100,
            temperature=0.3,
            repetition_penalty=1.1,
            do_sample=True,
            return_full_text=False
        )
        
    def invoke(self, query):
        if type(query) is dict: query = query.get("query", "")
        
        q_emb = self.embedding_model.encode([query])
        D, I = self.index.search(q_emb, k=2)
        
        context = "\n".join([self.docs[i] for i in I[0]])
        prompt = f"System: You are a Cyber AI. Answer using the context:\n{context}\n\nUser: {query}\n\nAssistant:"
        
        try:
            response = self.pipe(prompt)[0]['generated_text']
            return {"result": response.strip()}
        except Exception as e:
            return {"result": f"Model inference failed: {e}"}

def setup_rag_chain():
    try:
        return ManualRAGChain()
    except Exception as e:
        log.error(f"RAG Boot Failure: {e}")
        return None

if __name__ == "__main__":
    chain = setup_rag_chain()
    if chain:
        print("\n--- RAG Copilot Ready ---")
        res = chain.invoke("How do I mitigate Log4Shell?")
        print(f"\nA: {res['result']}")
