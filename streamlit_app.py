"""
Redrob · Intelligent Candidate Ranking System
India Runs Data & AI Challenge — Hackathon Submission
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io, os, re
from datetime import datetime
st.markdown("""
<style>
    img[src*="error"] { content: url("https://neural-x.streamlit.app/"); }
</style>
""", unsafe_allow_html=True)
st.set_page_config(
    page_title="Redrob · Candidate Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; background-color: #0a0a0f !important; color: #f0f0f8 !important; }
.stApp { background-color: #0a0a0f !important; }
[data-testid="stSidebar"] { background: #12121a !important; border-right: 1px solid #2a2a3a !important; }
[data-testid="stSidebar"] * { color: #f0f0f8 !important; }
h1,h2,h3,h4 { color: #f0f0f8 !important; font-family: 'Inter', sans-serif !important; }
[data-testid="stMetricValue"] { color: #a78bfa !important; font-family: 'JetBrains Mono', monospace !important; }
[data-testid="stMetricLabel"] { color: #8888aa !important; }
.stButton > button { background: linear-gradient(135deg,#6c63ff,#a78bfa) !important; color:white !important; border:none !important; border-radius:10px !important; font-weight:600 !important; padding:10px 24px !important; transition:all 0.2s !important; }
.stButton > button:hover { transform:translateY(-1px) !important; box-shadow:0 8px 24px rgba(108,99,255,0.35) !important; }
.stTextArea textarea, .stTextInput input { background:#12121a !important; color:#f0f0f8 !important; border:1px solid #2a2a3a !important; border-radius:10px !important; }
.stSelectbox > div > div { background:#12121a !important; border:1px solid #2a2a3a !important; border-radius:10px !important; color:#f0f0f8 !important; }
.stTabs [data-baseweb="tab-list"] { background:#12121a !important; border-bottom:1px solid #2a2a3a !important; gap:4px !important; padding:4px !important; border-radius:10px 10px 0 0 !important; }
.stTabs [data-baseweb="tab"] { background:transparent !important; color:#8888aa !important; border-radius:8px !important; font-weight:500 !important; }
.stTabs [aria-selected="true"] { background:rgba(108,99,255,0.2) !important; color:#a78bfa !important; }
[data-testid="stFileUploadDropzone"] { background:#12121a !important; border:2px dashed #2a2a3a !important; border-radius:12px !important; color:#8888aa !important; }
.stProgress > div > div { background:linear-gradient(90deg,#6c63ff,#a78bfa) !important; }
hr { border-color:#2a2a3a !important; }
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-thumb { background:#2a2a3a; border-radius:3px; }

.metric-card { background:#12121a; border:1px solid #2a2a3a; border-radius:14px; padding:20px; text-align:center; }
.metric-val { font-family:'JetBrains Mono',monospace; font-size:2rem; font-weight:700; background:linear-gradient(135deg,#6c63ff,#a78bfa); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.metric-label { font-size:0.75rem; color:#8888aa; margin-top:4px; text-transform:uppercase; letter-spacing:0.08em; }
.hero-banner { background:linear-gradient(135deg,rgba(108,99,255,0.15),rgba(167,139,250,0.08)); border:1px solid rgba(108,99,255,0.25); border-radius:20px; padding:32px 40px; margin-bottom:24px; text-align:center; }
.hero-banner h1 { font-size:2.2rem !important; font-weight:700 !important; letter-spacing:-0.03em !important; background:linear-gradient(135deg,#fff 40%,#a78bfa) !important; -webkit-background-clip:text !important; -webkit-text-fill-color:transparent !important; margin-bottom:8px !important; }
.hero-banner p { color:#8888aa; font-size:0.95rem; }
.info-banner { background:rgba(108,99,255,0.08); border:1px solid rgba(108,99,255,0.2); border-radius:12px; padding:14px 20px; margin:12px 0; font-size:0.85rem; color:#a78bfa; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
# ─── GLOBAL THEME CONFIGURATION (ENTIRE SCREEN) ───────────────────────────
# Add this single toggle in your sidebar to control the whole app's appearance
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=True, key="global_theme_toggle")

if dark_mode:
    # Global Dark Theme Injection
    st.markdown("""
        <style>
            /* App Background */
            .stApp {
                background-color: #0d0e12 !important;
                color: #ffffff !important;
            }
            /* Sidebar Background */
            [data-testid="stSidebar"] {
                background-color: #13151a !important;
            }
            /* Metric Cards globally across all tabs */
            .metric-card {
                background-color: #1e1e24 !important;
                border: 1px solid #2d2d34 !important;
                color: #ffffff !important;
                border-radius: 10px;
                padding: 15px;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            }
            .metric-val { font-size: 2rem; font-weight: 700; color: #a78bfa; }
            .metric-label { font-size: 0.85rem; color: #888aa0; margin-top: 4px; }
        </style>
    """, unsafe_allow_html=True)
else:
    # Global Light/Bright Theme Injection
    st.markdown("""
        <style>
            /* App Background */
            .stApp {
                background-color: #f8fafc !important;
                color: #0f172a !important;
            }
            /* Sidebar Background */
            [data-testid="stSidebar"] {
                background-color: #ffffff !important;
                border-right: 1px solid #e2e8f0;
            }
            /* Metric Cards globally across all tabs */
            .metric-card {
                background-color: #ffffff !important;
                border: 1px solid #e2e8f0 !important;
                color: #1e293b !important;
                border-radius: 10px;
                padding: 15px;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            }
            .metric-val { font-size: 2rem; font-weight: 700; color: #6d28d9; }
            .metric-label { font-size: 0.85rem; color: #64748b; margin-top: 4px; }
        </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
DEMO_JD = """Job Description: Senior AI Engineer — Founding Team
Company: Redrob AI (Series A AI-native talent intelligence platform)
Location: Pune/Noida, India | Hybrid
Experience Required: 5–9 years

MUST HAVE:
- Production experience with embeddings-based retrieval (sentence-transformers, BGE, E5, OpenAI embeddings) deployed to real users
- Production experience with vector databases (Pinecone, Weaviate, Qdrant, Milvus, FAISS, Elasticsearch)
- Strong Python — code quality matters
- Hands-on experience designing evaluation frameworks for ranking systems (NDCG, MRR, MAP, A/B test interpretation)
- Shipped at least one end-to-end ranking, search, or recommendation system at meaningful scale

NICE TO HAVE:
- LLM fine-tuning (LoRA, QLoRA, PEFT)
- Learning-to-rank models (XGBoost-based or neural)
- Prior exposure to HR-tech or marketplace products
- Open-source contributions in AI/ML

DISQUALIFIERS:
- Pure research background with no production deployment
- AI experience only from recent LLM wrapper projects (< 12 months)
- Entire career at consulting firms only (TCS, Infosys, Wipro, Accenture)
- Primary expertise in CV, speech, or robotics without NLP/IR exposure"""

DEMO_CANDIDATES = [
    {"id":"CAND_0005260","name":"Mira Ghosh","title":"Senior NLP Engineer","company":"Netflix","location":"Chennai, Tamil Nadu","years":5.2,"composite":88,"skill":85,"experience":90,"trajectory":88,"engagement":86.7,"semantic":0.828,"flag":"none","reasoning":"Strong NLP and embeddings background at Netflix with demonstrated production deployment. Career trajectory shows consistent growth in retrieval systems. High engagement — actively looking.","strengths":["Production embeddings retrieval","Hybrid search experience","Fast-paced startup fit"],"gaps":["Limited explicit vector DB experience","No distributed systems mention"],"resume":"Mira Ghosh | Senior NLP Engineer @ Netflix\n5+ years building production NLP and search systems. Led embedding-based retrieval pipeline serving 50M+ users. Expertise in sentence-transformers, FAISS, Elasticsearch. Designed offline evaluation frameworks using NDCG and MRR.\n\nSkills: Python, PyTorch, sentence-transformers, FAISS, Elasticsearch, HuggingFace, NDCG, MRR, A/B testing, Kafka, Spark"},
    {"id":"CAND_0018499","name":"Aarav Trivedi","title":"Senior ML Engineer","company":"Zomato","location":"Noida, Uttar Pradesh","years":7.2,"composite":88,"skill":85,"experience":90,"trajectory":80,"engagement":78.5,"semantic":0.855,"flag":"none","reasoning":"Extensive production ML at Zomato — a high-scale product company. Search and retrieval work is directly relevant. Located in Noida which is ideal for this role.","strengths":["Extensive production ML","Embeddings & hybrid retrieval","Evaluation frameworks"],"gaps":["No mention of sentence-transformers specifically","No open-source contributions listed"],"resume":"Aarav Trivedi | Senior ML Engineer @ Zomato\n7+ years in applied ML. Built Zomato's restaurant search ranking system (BGE embeddings + BM25 hybrid). Shipped real-time recommendation engine serving 3M daily active users.\n\nSkills: Python, BGE, BM25, Qdrant, Spark, XGBoost, LambdaMART, NDCG, A/B testing"},
    {"id":"CAND_0039754","name":"Mira Banerjee","title":"Senior Applied Scientist","company":"Meta","location":"Indore, Madhya Pradesh","years":16.2,"composite":88,"skill":85,"experience":90,"trajectory":80,"engagement":75.3,"semantic":0.865,"flag":"none","reasoning":"Very strong technical background from Meta. 16 years is above the target range but depth in search/ranking is exceptional.","strengths":["Search retrieval & ranking expert","NDCG@10 track record","Embedding model fine-tuning"],"gaps":["Above seniority range (16y)","No mention of specific open-source vector DBs"],"resume":"Mira Banerjee | Senior Applied Scientist @ Meta\n16 years in search & ranking. Improved NDCG@10 by 12% on Facebook Search. Expert in embedding model selection, fine-tuning, and evaluation.\n\nSkills: Python, FAISS, DPR, ColBERT, sentence-transformers, NDCG, MRR, MAP, PyTorch, Spark"},
    {"id":"CAND_0081846","name":"Arjun Khanna","title":"Lead AI Engineer","company":"Razorpay","location":"Jaipur, Rajasthan","years":6.7,"composite":88,"skill":85,"experience":90,"trajectory":80,"engagement":78.2,"semantic":0.826,"flag":"none","reasoning":"Lead AI at Razorpay — a high-growth fintech product company. Strong embedding retrieval and vector DB background. Open to relocation.","strengths":["Production embeddings retrieval","Vector database experience","Large-scale system shipping"],"gaps":["No LLM fine-tuning evidence","No Pinecone/Weaviate/Qdrant mentioned"],"resume":"Arjun Khanna | Lead AI Engineer @ Razorpay\n6.7 years building product search systems. Migrated search from BM25 to dense retrieval (sentence-transformers + FAISS), reducing P90 latency by 40%.\n\nSkills: Python, sentence-transformers, FAISS, Milvus, PyTorch, XGBoost, NDCG, A/B testing, Kubernetes"},
    {"id":"CAND_0002025","name":"Ira Dalal","title":"Senior AI Engineer","company":"Apple","location":"Trivandrum, Kerala","years":5.9,"composite":87,"skill":85,"experience":90,"trajectory":88,"engagement":72.2,"semantic":0.890,"flag":"none","reasoning":"Apple background gives strong signal of production-grade quality bar. Embeddings and Pinecone work well-evidenced. Lower engagement — may need active outreach.","strengths":["Embeddings retrieval & vector DBs","End-to-end system shipping","Hybrid retrieval & LTR"],"gaps":["Lower engagement score","No explicit mention of FAISS"],"resume":"Ira Dalal | Senior AI Engineer @ Apple\nBuilt Siri's intent retrieval system using E5-large and Pinecone. 5.9 years across Apple and Flipkart.\n\nSkills: Python, E5, Pinecone, Weaviate, LambdaMART, NDCG, MRR, PyTorch, Kubernetes"},
    {"id":"CAND_0006418","name":"Rahul Mukherjee","title":"ML Engineer","company":"Verloop.io","location":"Gurgaon, Haryana","years":5.7,"composite":88,"skill":85,"experience":90,"trajectory":88,"engagement":76.4,"semantic":0.814,"flag":"none","reasoning":"Verloop.io is a product company in AI — strong signal. Good match across embeddings, vector DBs, and evaluation. Gurgaon location is ideal for Noida office.","strengths":["Embeddings retrieval","Vector DBs","Evaluation frameworks","Recent shipping experience"],"gaps":["No LLM fine-tuning","No specific vector DB brand mentioned"],"resume":"Rahul Mukherjee | ML Engineer @ Verloop.io\n5.7 years in conversational AI and semantic search. Built customer support search using sentence-transformers + Qdrant.\n\nSkills: Python, sentence-transformers, Qdrant, Elasticsearch, NDCG, MAP, FastAPI, Docker, PyTorch"},
    {"id":"CAND_0092278","name":"Ananya Arora","title":"Senior NLP Engineer","company":"Microsoft","location":"Pune, Maharashtra","years":6.8,"composite":88,"skill":85,"experience":90,"trajectory":80,"engagement":53.9,"semantic":0.864,"flag":"minor_concern","reasoning":"Strong NLP background at Microsoft but minor inconsistencies in claimed embedding tools. Low engagement score is a concern — may not be actively looking.","strengths":["NLP and embedding expertise","Production ML systems","Evaluation frameworks"],"gaps":["Low engagement (53.9) — may not be actively looking","Skills partially unverified against work history"],"resume":"Ananya Arora | Senior NLP Engineer @ Microsoft\n6.8 years in NLP. Works on Azure Cognitive Search team. Claims sentence-transformers/BGE but primarily used Azure proprietary APIs.\n\nSkills: Python, Azure Cognitive Search, sentence-transformers (claimed), BGE (claimed), NDCG, MRR"},
    {"id":"CAND_0071974","name":"Sai Verma","title":"Senior AI Engineer","company":"Netflix","location":"Vizag, Andhra Pradesh","years":7.8,"composite":82,"skill":85,"experience":90,"trajectory":80,"engagement":74.5,"semantic":0.873,"flag":"minor_concern","reasoning":"Good overall background but skill claims are broader than evidenced — lists CV, Speech, Object Detection alongside NLP/retrieval. Needs core retrieval verification.","strengths":["Production embeddings retrieval","Vector DB experience","Strong Python"],"gaps":["Skill list too broad — CV, Speech, Object Detection listed (possible padding)","Core retrieval depth unverified"],"resume":"Sai Verma | Senior AI Engineer @ Netflix\n7.8 years across domains. Lists: BM25, CNN, RAG, Speech Recognition, Object Detection, sentence-transformers, Pinecone — overly broad.\n\nSkills: Python, sentence-transformers, Pinecone, CNN, LSTM, Speech Recognition, Object Detection"},
    {"id":"CAND_0074225","name":"Kabir Agarwal","title":"ML Engineer","company":"Unacademy","location":"Vizag, Andhra Pradesh","years":4.3,"composite":78,"skill":75,"experience":80,"trajectory":85,"engagement":73.8,"semantic":0.789,"flag":"none","reasoning":"Below minimum experience (4.3y vs 5y required) but shows strong growth trajectory. Recommendation systems work is relevant. Would need retrieval depth verified in interview.","strengths":["Strong ML background","Recommendation systems","Good growth trajectory"],"gaps":["Below minimum experience (4.3y vs 5y required)","No direct embeddings retrieval evidence","No vector DB experience evidenced"],"resume":"Kabir Agarwal | ML Engineer @ Unacademy\n4.3 years in ML. Built course recommendation engine using collaborative filtering. Recent LangChain project (< 6 months).\n\nSkills: Python, scikit-learn, collaborative filtering, LangChain (recent), PyTorch"},
    {"id":"CAND_0999001","name":"Priya Sharma","title":"Senior Data Scientist","company":"Accenture","location":"Bangalore, Karnataka","years":8.0,"composite":30,"skill":25,"experience":40,"trajectory":35,"engagement":45.0,"semantic":0.41,"flag":"major_concern","reasoning":"DISQUALIFIED: Entire career at Accenture (consulting firm only). Skills are analytics/BI focused, not ML engineering. Classic keyword-padded resume — lists AI/ML/Python but actual work is Tableau/Power BI reporting.","strengths":["8 years total experience"],"gaps":["Entire career at consulting firm only","No product company experience","Skills are BI/analytics, not ML engineering","Keyword padding: AI/ML listed but actual work is Excel/Tableau reporting"],"resume":"Priya Sharma | Senior Data Scientist @ Accenture\n8 years at Accenture. Actual work: Tableau, Power BI, Excel, SQL for BFSI banking clients.\n\nClaimed Skills: Python, ML, AI, Deep Learning, TensorFlow (unverified — actual stack is Tableau/Excel)"},
]

REAL_RESULTS = [
    {"rank":1,"id":"CAND_0005260","name":"Mira Ghosh","title":"Senior NLP Engineer","company":"Netflix","location":"Chennai, Tamil Nadu","years":5.2,"composite":88,"skill":85,"experience":90,"trajectory":88,"engagement":86.7,"semantic":0.828,"flag":"none","reasoning":"Strong NLP and embeddings background. Production ML track record. High engagement score.","strengths":["Strong NLP experience","Embeddings & hybrid search","Fast-paced startup fit"],"gaps":["Limited vector DB evidence","No distributed systems"]},
    {"rank":2,"id":"CAND_0018499","name":"Aarav Trivedi","title":"Senior ML Engineer","company":"Zomato","location":"Noida, Uttar Pradesh","years":7.2,"composite":88,"skill":85,"experience":90,"trajectory":80,"engagement":78.5,"semantic":0.855,"flag":"none","reasoning":"Extensive production ML at Zomato. Search/retrieval directly relevant. Ideal Noida location.","strengths":["Production ML systems","Embeddings & hybrid retrieval","Evaluation frameworks"],"gaps":["No sentence-transformers specifically","No open-source contributions"]},
    {"rank":3,"id":"CAND_0039754","name":"Mira Banerjee","title":"Senior Applied Scientist","company":"Meta","location":"Indore, Madhya Pradesh","years":16.2,"composite":88,"skill":85,"experience":90,"trajectory":80,"engagement":75.3,"semantic":0.865,"flag":"none","reasoning":"Exceptional depth in search/ranking from Meta. Above seniority range but technically outstanding.","strengths":["Search/retrieval/ranking expert","NDCG@10 track record","Embedding fine-tuning"],"gaps":["Above experience range","No specific open-source vector DB"]},
    {"rank":4,"id":"CAND_0081846","name":"Arjun Khanna","title":"Lead AI Engineer","company":"Razorpay","location":"Jaipur, Rajasthan","years":6.7,"composite":88,"skill":85,"experience":90,"trajectory":80,"engagement":78.2,"semantic":0.826,"flag":"none","reasoning":"Lead AI at high-growth fintech. Strong retrieval and vector DB work. Open to relocation.","strengths":["Production embeddings retrieval","Vector database experience","Large-scale system shipping"],"gaps":["No LLM fine-tuning","No Pinecone/Weaviate"]},
    {"rank":5,"id":"CAND_0002025","name":"Ira Dalal","title":"Senior AI Engineer","company":"Apple","location":"Trivandrum, Kerala","years":5.9,"composite":87,"skill":85,"experience":90,"trajectory":88,"engagement":72.2,"semantic":0.890,"flag":"none","reasoning":"Apple quality bar. Embeddings and Pinecone work well-evidenced. Lower engagement needs active outreach.","strengths":["Embeddings retrieval & vector DBs","End-to-end system shipping","Hybrid retrieval & LTR"],"gaps":["Lower engagement score","No explicit FAISS mention"]},
    {"rank":6,"id":"CAND_0006418","name":"Rahul Mukherjee","title":"ML Engineer","company":"Verloop.io","location":"Gurgaon, Haryana","years":5.7,"composite":88,"skill":85,"experience":90,"trajectory":88,"engagement":76.4,"semantic":0.814,"flag":"none","reasoning":"Product-company AI background. Qdrant + evaluation pipeline work directly relevant. Ideal location.","strengths":["Embeddings retrieval","Vector DBs","Evaluation frameworks","Recent shipping"],"gaps":["No LLM fine-tuning","No specific vector DB brand"]},
    {"rank":7,"id":"CAND_0008425","name":"Myra Krishnan","title":"Senior NLP Engineer","company":"Ola","location":"Kolkata, West Bengal","years":7.8,"composite":88,"skill":85,"experience":90,"trajectory":80,"engagement":60.3,"semantic":0.847,"flag":"none","reasoning":"Strong production NLP at Ola. Search and retrieval background solid. Lower engagement — needs outreach.","strengths":["Production embeddings retrieval","Strong Python","End-to-end shipping"],"gaps":["No explicit vector DB mention","Low engagement score"]},
    {"rank":8,"id":"CAND_0024620","name":"Avni Rao","title":"AI Engineer","company":"PharmEasy","location":"New York","years":5.9,"composite":83,"skill":85,"experience":80,"trajectory":85,"engagement":70.7,"semantic":0.824,"flag":"none","reasoning":"Good embeddings and recommendation background. New York location — needs India relocation confirmation.","strengths":["Embeddings retrieval","Vector databases","Recommendation systems"],"gaps":["Location: New York","No LLM fine-tuning"]},
    {"rank":9,"id":"CAND_0052328","name":"Vikram Banerjee","title":"Recommendation Systems Engineer","company":"Amazon","location":"Pune, Maharashtra","years":6.5,"composite":82,"skill":78,"experience":85,"trajectory":80,"engagement":76.2,"semantic":0.821,"flag":"none","reasoning":"Amazon recommendation systems background. Pune location ideal. Needs verification of retrieval depth.","strengths":["Recommendation systems","Product company (Amazon)","Good educational background"],"gaps":["Limited embeddings retrieval mention","No vector DB evidence"]},
    {"rank":10,"id":"CAND_0030953","name":"Dhruv Hegde","title":"Search Engineer","company":"Nykaa","location":"Chennai, Tamil Nadu","years":7.8,"composite":82,"skill":85,"experience":80,"trajectory":75,"engagement":73.3,"semantic":0.819,"flag":"none","reasoning":"Direct search engineering background. Nykaa is a product company. LLM fine-tuning gap is acceptable.","strengths":["Embedding-based retrieval","Vector databases","NLP experience"],"gaps":["No LLM fine-tuning","No distributed systems"]},
    {"rank":11,"id":"CAND_0069905","name":"Nisha Bansal","title":"Applied ML Engineer","company":"Sarvam AI","location":"Bhubaneswar, Odisha","years":6.6,"composite":82,"skill":78,"experience":85,"trajectory":80,"engagement":71.9,"semantic":0.823,"flag":"none","reasoning":"Sarvam AI is a strong Indian AI product company. Long-term applied ML experience.","strengths":["Embeddings retrieval","Vector databases","Long-term applied ML"],"gaps":["No distributed systems","No open-source contributions"]},
    {"rank":12,"id":"CAND_0083879","name":"Mira Reddy","title":"ML Engineer","company":"Ola","location":"Noida, Uttar Pradesh","years":7.1,"composite":83,"skill":85,"experience":80,"trajectory":85,"engagement":68.7,"semantic":0.795,"flag":"none","reasoning":"Noida location ideal. Product company ML background. Needs Pinecone/Qdrant confirmation.","strengths":["Embeddings and vector DBs","NLP and applied AI","Production shipping"],"gaps":["No Pinecone/Qdrant evidence","No distributed systems"]},
    {"rank":13,"id":"CAND_0055905","name":"Anika Rao","title":"Senior ML Engineer","company":"Flipkart","location":"London","years":8.1,"composite":80,"skill":75,"experience":85,"trajectory":80,"engagement":75.9,"semantic":0.851,"flag":"none","reasoning":"Flipkart is strong signal — high-scale search/retrieval experience. London location needs India return confirmation.","strengths":["Embeddings retrieval","Vector DB & hybrid search","End-to-end shipping"],"gaps":["Location: London","No LLM fine-tuning"]},
    {"rank":14,"id":"CAND_0011162","name":"Nisha Sharma","title":"Recommendation Systems Engineer","company":"upGrad","location":"Coimbatore, Tamil Nadu","years":5.8,"composite":82,"skill":78,"experience":85,"trajectory":80,"engagement":71.1,"semantic":0.803,"flag":"none","reasoning":"upGrad product company. Retrieval migration project shows hands-on production experience.","strengths":["Embeddings retrieval","Vector databases","Strong ML background"],"gaps":["No FAISS evidence","No LLM fine-tuning"]},
    {"rank":15,"id":"CAND_0086022","name":"Dhruv Naidu","title":"Senior Applied Scientist","company":"Sarvam AI","location":"Kolkata, West Bengal","years":5.3,"composite":82,"skill":78,"experience":85,"trajectory":80,"engagement":64.0,"semantic":0.832,"flag":"none","reasoning":"Sarvam AI background strong. Good track record in shipping systems. Lower engagement noted.","strengths":["Embeddings retrieval","Vector databases","End-to-end shipping"],"gaps":["No hybrid search systems","No LLM fine-tuning"]},
    {"rank":16,"id":"CAND_0078492","name":"Aadhya Vora","title":"Recommendation Systems Engineer","company":"Verloop.io","location":"Kochi, Kerala","years":5.1,"composite":82,"skill":85,"experience":80,"trajectory":75,"engagement":66.9,"semantic":0.798,"flag":"none","reasoning":"FAISS and sentence-transformers at Verloop.io directly relevant. Semantic search infrastructure is on-point.","strengths":["Embeddings retrieval","FAISS & sentence-transformers","Recommendation systems"],"gaps":["No Pinecone/Qdrant","No LLM fine-tuning"]},
    {"rank":17,"id":"CAND_0053695","name":"Sanjay Sharma","title":"Recommendation Systems Engineer","company":"Meesho","location":"Bhubaneswar, Odisha","years":5.8,"composite":82,"skill":78,"experience":85,"trajectory":80,"engagement":54.1,"semantic":0.808,"flag":"none","reasoning":"Meesho recommendation systems work is relevant. Minor skill inconsistencies noted. Low engagement.","strengths":["Embeddings retrieval","Vector DB & hybrid search","Recommendation systems"],"gaps":["Skill inconsistencies","No LLM fine-tuning"]},
    {"rank":18,"id":"CAND_0074225","name":"Kabir Agarwal","title":"ML Engineer","company":"Unacademy","location":"Vizag, Andhra Pradesh","years":4.3,"composite":78,"skill":75,"experience":80,"trajectory":85,"engagement":73.8,"semantic":0.789,"flag":"none","reasoning":"Below experience minimum (4.3y). Good growth trajectory. Needs deeper retrieval verification in interview.","strengths":["ML background","Recommendation systems","Growth trajectory"],"gaps":["Below min experience","No embeddings retrieval evidence","No vector DB"]},
    {"rank":19,"id":"CAND_0092278","name":"Ananya Arora","title":"Senior NLP Engineer","company":"Microsoft","location":"Pune, Maharashtra","years":6.8,"composite":88,"skill":85,"experience":90,"trajectory":80,"engagement":53.9,"semantic":0.864,"flag":"minor_concern","reasoning":"Microsoft background strong but minor skill inconsistencies — sentence-transformers claimed but Azure APIs primarily used. Low engagement score is a concern.","strengths":["NLP & embedding expertise","Production ML systems","Evaluation frameworks"],"gaps":["Low engagement (53.9)","Skills partially unverified"]},
    {"rank":20,"id":"CAND_0068351","name":"Aadhya Iyer","title":"Lead AI Engineer","company":"Sarvam AI","location":"Delhi, Delhi","years":6.4,"composite":82,"skill":78,"experience":90,"trajectory":85,"engagement":81.4,"semantic":0.864,"flag":"minor_concern","reasoning":"Overstates some skills not fully supported by recent roles. Good potential but needs interview verification.","strengths":["ML background","Production systems","Fast-paced adaptability"],"gaps":["Limited retrieval evidence","No explicit vector DB"]},
    {"rank":21,"id":"CAND_0071974","name":"Sai Verma","title":"Senior AI Engineer","company":"Netflix","location":"Vizag, Andhra Pradesh","years":7.8,"composite":82,"skill":85,"experience":90,"trajectory":80,"engagement":74.5,"semantic":0.873,"flag":"minor_concern","reasoning":"Broad skill list raises concerns — CV, Speech, Object Detection alongside retrieval suggests keyword padding. Core retrieval depth needs verification.","strengths":["Embeddings retrieval","Vector DB","Strong Python"],"gaps":["Skill list too broad — possible padding","Core retrieval depth unverified"]},
    {"rank":22,"id":"CAND_0088025","name":"Amit Arora","title":"Staff ML Engineer","company":"Yellow.ai","location":"Jaipur, Rajasthan","years":8.6,"composite":82,"skill":78,"experience":90,"trajectory":85,"engagement":80.1,"semantic":0.840,"flag":"minor_concern","reasoning":"Search/retrieval background exists but gaps in specific technologies. Minor inconsistencies in experience claims.","strengths":["Embeddings retrieval","Hybrid search","Evaluation frameworks"],"gaps":["No LLM fine-tuning","No distributed systems"]},
    {"rank":23,"id":"CAND_0093193","name":"Aarohi Bose","title":"Senior ML Engineer","company":"Niramai","location":"Bangalore, Karnataka","years":7.9,"composite":82,"skill":85,"experience":90,"trajectory":80,"engagement":73.4,"semantic":0.854,"flag":"minor_concern","reasoning":"ML background strong but direct evidence for vector databases and retrieval systems is lacking.","strengths":["Strong ML experience","Full-stack experience","Ships over research"],"gaps":["No vector DB evidence","Limited retrieval evidence"]},
    {"rank":24,"id":"CAND_0093912","name":"Advik Sethi","title":"Senior Data Scientist","company":"Razorpay","location":"Chandigarh, Chandigarh","years":5.3,"composite":82,"skill":78,"experience":85,"trajectory":80,"engagement":75.0,"semantic":0.843,"flag":"minor_concern","reasoning":"Elasticsearch experience is relevant but embeddings and vector DB claims lack direct evidence in work history.","strengths":["ML background","Elasticsearch","RAG & eval frameworks"],"gaps":["No embeddings evidence","No vector DB evidence"]},
    {"rank":25,"id":"CAND_0040820","name":"Ayaan Vora","title":"Senior Software Engineer (ML)","company":"Meesho","location":"Ahmedabad, Gujarat","years":5.5,"composite":82,"skill":78,"experience":85,"trajectory":80,"engagement":64.6,"semantic":0.801,"flag":"minor_concern","reasoning":"ML engineering skills present but lacks direct evidence of production retrieval systems.","strengths":["NLP expertise","ML engineering","Long-term industry"],"gaps":["No embeddings retrieval","No vector DB","No end-to-end shipping evidence"]},
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def compute_score(c, weights):
    wc = weights["composite"] / 100
    ws = weights["semantic"] / 100
    we = weights["engagement"] / 100
    sem = c.get("semantic", 0.83)
    sem_pct = sem * 100 if sem <= 1 else sem
    score = wc * c["composite"] + ws * sem_pct + we * c["engagement"]
    if c["flag"] == "major_concern": score *= 0.7
    elif c["flag"] == "minor_concern": score *= 0.9
    return round(score, 1)

def flag_badge(flag):
    return {"none": "🟢 Verified", "minor_concern": "🟡 Minor Flag", "major_concern": "🔴 Major Flag"}.get(flag, "🟢 Verified")

def status_label(score, flag):
    if flag == "major_concern": return "⛔ Rejected"
    if score >= 83: return "✅ Shortlisted"
    if score >= 75: return "👁 Review"
    return "⛔ Rejected"

def make_df(candidates, weights):
    rows = []
    for c in candidates:
        score = compute_score(c, weights)
        rows.append({
            "Name": c["name"], "Title": c["title"], "Company": c["company"],
            "Location": c["location"], "Exp (yrs)": c["years"],
            "Final Score": score, "Composite": c["composite"],
            "Skill Match": c["skill"], "Experience": c["experience"],
            "Trajectory": c["trajectory"], "Engagement": round(c["engagement"], 1),
            "Authenticity": flag_badge(c["flag"]),
            "Status": status_label(score, c["flag"]),
            "_flag": c["flag"],
        })
    df = pd.DataFrame(rows).sort_values("Final Score", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df

def extract_text(f):
    try:
        if f.name.endswith(".txt"):
            return f.read().decode("utf-8", errors="ignore")
        if f.name.endswith(".pdf"):
            try:
                import pdfplumber
                with pdfplumber.open(f) as pdf:
                    return "\n".join(p.extract_text() or "" for p in pdf.pages)
            except: return f.read().decode("utf-8", errors="ignore")
        if f.name.endswith(".docx"):
            try:
                from docx import Document
                return "\n".join(p.text for p in Document(f).paragraphs if p.text.strip())
            except: return f.read().decode("utf-8", errors="ignore")
    except: return ""

def heuristic_score(name, text, jd, weights):
    jd_kw = set(re.findall(r'\b\w{4,}\b', jd.lower()))
    res_kw = set(re.findall(r'\b\w{4,}\b', text.lower()))
    power = {"embeddings","retrieval","vector","faiss","pinecone","qdrant","milvus","elasticsearch",
             "sentence","transformers","bge","ndcg","mrr","ranking","recommendation","production"}
    overlap = jd_kw & res_kw
    skill = min(95, int(len(overlap)/max(len(jd_kw),1)*250) + len(power & res_kw)*3)
    yr = re.search(r'(\d+)\s*\+?\s*year', text.lower())
    years = float(yr.group(1)) if yr else 3.0
    exp = min(95, int(years/8*80+20))
    composite = int(skill*0.6 + exp*0.4)
    return {"id":f"UP_{name[:5].upper()}", "name":name, "title":"Uploaded", "company":"Uploaded",
            "location":"—", "years":years, "composite":composite, "skill":skill, "experience":exp,
            "trajectory":75, "engagement":70.0, "semantic":0.75, "flag":"none",
            "reasoning":f"Heuristic: {len(overlap)} JD keyword matches, {len(power&res_kw)} power-term hits.",
            "strengths":[f"{len(overlap)} JD keywords matched"], "gaps":["Full LLM eval not run"]}

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
if "candidates" not in st.session_state: st.session_state.candidates = []
if "jd_text" not in st.session_state: st.session_state.jd_text = ""
if "mode" not in st.session_state: st.session_state.mode = None   # "demo" | "real" | "upload"

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ Redrob Intelligence")
    st.markdown("*India Runs AI Challenge*")
    st.divider()

    st.markdown("**Score Weightage**")
    st.caption("Rebalance the ranking formula in real time")
    w_composite  = st.slider("🧠 LLM Composite",      0, 100, 70, 5)
    w_semantic   = st.slider("🔍 Semantic Similarity", 0, 100, 20, 5)
    w_engagement = st.slider("📊 Engagement Signal",   0, 100, 10, 5)
    total = w_composite + w_semantic + w_engagement
    if total != 100: st.warning(f"Weights sum to {total}% (ideally 100%)")
    weights = {"composite": w_composite, "semantic": w_semantic, "engagement": w_engagement}

    st.divider()
    st.markdown("**Filter by Status**")
    show_short  = st.checkbox("✅ Shortlisted", True)
    show_review = st.checkbox("👁 Review",       True)
    show_reject = st.checkbox("⛔ Rejected",     False)

    st.divider()
    st.markdown("**Pipeline Stages**")
    st.markdown("""
    1. 📄 Role intelligence (LLM)
    2. 🧠 Candidate profiling
    3. ⚡ Semantic pre-filter (BGE)
    4. 🔬 Deep LLM evaluation
    5. 🏆 MMR diversity ranking
    """)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <h1>⚡ Redrob · Candidate Intelligence</h1>
  <p>Ranks candidates the way a great recruiter would — not by matching keywords, but by understanding who genuinely fits.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TOP BUTTONS
# ─────────────────────────────────────────────────────────────────────────────
b1, b2, b3 = st.columns(3)

with b1:
    if st.button("⚡ Load Demo Hackathon Data", use_container_width=True):
        import time
        p = st.progress(0)
        for pct, msg in [(20,"Parsing JD..."),(50,"Loading candidates..."),(80,"Scoring..."),(100,"Done!")]:
            time.sleep(0.18); p.progress(pct, text=msg)
        time.sleep(0.15); p.empty()
        st.session_state.candidates = DEMO_CANDIDATES
        st.session_state.jd_text    = DEMO_JD
        st.session_state.mode       = "demo"
        st.success(f"✅ Demo loaded — {len(DEMO_CANDIDATES)} candidates (10 profiles with intentional skill gaps)")
        st.rerun()

with b2:
    if st.button("🏆 Load Real Results (100K Run)", use_container_width=True):
        import time
        p = st.progress(0)
        for pct, msg in [(30,"Loading pipeline output..."),(70,"Reading 100K candidate evaluations..."),(100,"Done!")]:
            time.sleep(0.2); p.progress(pct, text=msg)
        time.sleep(0.15); p.empty()
        st.session_state.candidates = REAL_RESULTS
        st.session_state.jd_text    = DEMO_JD
        st.session_state.mode       = "real"
        st.success("✅ Real results loaded — top 25 shortlisted from 100,000 candidates evaluated on GPU cluster")
        st.rerun()

with b3:
    if st.button("🔄 Clear & Start Fresh", use_container_width=True):
        st.session_state.candidates = []
        st.session_state.jd_text    = ""
        st.session_state.mode       = None
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Job Description",
    "📂 Upload Candidates",
    "🏆 Leaderboard",
    "📊 Analytics",
    "🔍 Candidate Profiles",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — JD
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Job Description & Role Configuration")
    c1, c2 = st.columns([3, 2])
    with c1:
        jd_in = st.text_area("Paste full job description", value=st.session_state.jd_text, height=380,
                              placeholder="Paste the complete JD here — requirements, must-haves, disqualifiers...")
        if jd_in != st.session_state.jd_text:
            st.session_state.jd_text = jd_in
    with c2:
        st.markdown("**Role Metadata**")
        st.selectbox("Role", ["Senior AI Engineer","ML Engineer","Data Scientist","Applied Scientist","NLP Engineer","Search Engineer"])
        st.selectbox("Department", ["AI / ML","Engineering","Data","Product"])
        st.selectbox("Min Experience", ["3 years","5 years","7 years","10 years"])
        st.selectbox("Location", ["Pune / Noida","Bangalore","Mumbai","Remote","Any India"])
        st.divider()
        st.markdown("**Required Skills**")
        st.multiselect("Must-have skills",
            ["Embeddings Retrieval","Vector Databases","Python","NDCG/MRR","LLM Fine-tuning",
             "Recommendation Systems","Hybrid Search","FAISS","Pinecone","A/B Testing","Learning to Rank"],
            default=["Embeddings Retrieval","Vector Databases","Python","NDCG/MRR"])
        st.divider()
        st.markdown("**Disqualifiers**")
        st.markdown("❌ Pure research only\n❌ Consulting-firm-only career\n❌ No code for 18+ months\n❌ CV/Speech/Robotics only")
    if st.session_state.jd_text:
        st.info(f"📋 JD loaded — {len(st.session_state.jd_text.split())} words")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Upload
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Candidate Resume Batch Uploader")
    files = st.file_uploader("Drag & drop resumes (PDF, DOCX, TXT)", type=["pdf","docx","txt"], accept_multiple_files=True)
    if files:
        if not st.session_state.jd_text:
            st.warning("⚠️ Load a Job Description first (Tab 1 or Demo button)")
        else:
            new_cands = []
            bar = st.progress(0); status = st.empty()
            for i, f in enumerate(files):
                bar.progress(int(i/len(files)*100), text=f"Processing {f.name}...")
                status.markdown(f"Extracting `{f.name}` ({i+1}/{len(files)})")
                text = extract_text(f)
                name = f.name.rsplit(".",1)[0].replace("_"," ").title()
                new_cands.append(heuristic_score(name, text, st.session_state.jd_text, weights))
            bar.progress(100, text="✅ Done!"); status.empty()
            existing = {c["id"] for c in st.session_state.candidates}
            added = [c for c in new_cands if c["id"] not in existing]
            st.session_state.candidates.extend(added)
            st.session_state.mode = "upload"
            st.success(f"✅ {len(added)} resumes added to ranking pool")
            for c in new_cands:
                with st.expander(f"📄 {c['name']} — Score preview"):
                    st.json({"skill_match": c["skill"], "experience": c["experience"], "years": c["years"], "reasoning": c["reasoning"]})
    else:
        st.markdown("""<div style="text-align:center;padding:60px 40px;border:2px dashed #2a2a3a;border-radius:16px;color:#8888aa;">
            <div style="font-size:3rem;margin-bottom:16px;">📂</div>
            <div style="font-size:1.1rem;font-weight:600;margin-bottom:8px;">Drop resumes here</div>
            <div style="font-size:0.85rem;">PDF · DOCX · TXT · Multiple files at once</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Leaderboard
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🏆 Candidate Ranking Leaderboard")

    if not st.session_state.candidates:
        st.info("👆 Click **Load Demo Hackathon Data** or **Load Real Results** above")
    else:
        df_all = make_df(st.session_state.candidates, weights)

        # ── Context banner ──────────────────────────────────────────────────
        if st.session_state.mode == "real":
            st.markdown("""<div class="info-banner">
            🏆 <strong>Real GPU Run Results</strong> — Pipeline evaluated 100,000 candidates using BAAI/bge-small embeddings
            for semantic filtering, then deep LLM scoring on top 800. These are the <strong>25 best matches</strong> shortlisted.
            </div>""", unsafe_allow_html=True)
        elif st.session_state.mode == "demo":
            st.markdown("""<div class="info-banner">
            ⚡ <strong>Demo Mode</strong> — 10 pre-loaded candidates with intentional skill gaps to showcase
            the full ranking pipeline, scoring dimensions, and authenticity detection.
            </div>""", unsafe_allow_html=True)

        # ── Status filter ────────────────────────────────────────────────────
        statuses = []
        if show_short:  statuses.append("✅ Shortlisted")
        if show_review: statuses.append("👁 Review")
        if show_reject: statuses.append("⛔ Rejected")
        df = df_all[df_all["Status"].isin(statuses)] if statuses else df_all.copy()

        # ── KPI cards ────────────────────────────────────────────────────────
        # For real mode: show 100K total, 25 shortlisted
        # For demo mode: show actual counts
        if st.session_state.mode == "real":
            total_pool   = 100_000
            shortlisted  = 25
            top_score    = df["Final Score"].max()
            avg_score    = df["Final Score"].mean()
            verified     = len(df[df["Authenticity"] == "🟢 Verified"])
        else:
            total_pool   = len(df)
            shortlisted  = len(df[df["Status"] == "✅ Shortlisted"])
            top_score    = df["Final Score"].max() if len(df) else 0
            avg_score    = df["Final Score"].mean() if len(df) else 0
            verified     = len(df[df["Authenticity"] == "🟢 Verified"])

        m1,m2,m3,m4,m5 = st.columns(5)
        m1.markdown(f'<div class="metric-card"><div class="metric-val">{total_pool:,}</div><div class="metric-label">Total Evaluated</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#22c55e">{shortlisted}</div><div class="metric-label">Shortlisted</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-card"><div class="metric-val">{top_score:.1f}</div><div class="metric-label">Top Score</div></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="metric-card"><div class="metric-val">{avg_score:.1f}</div><div class="metric-label">Avg Score</div></div>', unsafe_allow_html=True)
        m5.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#22c55e">{verified}</div><div class="metric-label">Verified</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Sort & paginate ──────────────────────────────────────────────────
        sort_opts = [c for c in ["Final Score","Skill Match","Experience","Trajectory","Engagement","Exp (yrs)"] if c in df.columns]
        sc1, sc2 = st.columns([2,4])
        with sc1: sort_col = st.selectbox("Sort by", sort_opts)
        df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
        df["Rank"] = range(1, len(df)+1)

        per_page = 15
        total_pages = max(1, int(np.ceil(len(df)/per_page)))
        pg1, pg2 = st.columns([1,5])
        with pg1: page = st.number_input("Page", 1, total_pages, 1)
        with pg2: st.markdown(f"<div style='padding-top:28px;color:#8888aa;'>Showing {(page-1)*per_page+1}–{min(page*per_page,len(df))} of {len(df)}</div>", unsafe_allow_html=True)

        disp_cols = ["Rank","Name","Title","Company","Location","Exp (yrs)","Final Score","Skill Match","Experience","Trajectory","Engagement","Authenticity","Status"]
        disp_cols = [c for c in disp_cols if c in df.columns]

        st.dataframe(
            df[disp_cols].iloc[(page-1)*per_page : page*per_page],
            use_container_width=True, height=460, hide_index=True,
            column_config={
                "Rank":        st.column_config.NumberColumn("🏅", width="small"),
                "Final Score": st.column_config.ProgressColumn("Final Score", min_value=0, max_value=100, format="%.1f"),
                "Skill Match": st.column_config.ProgressColumn("Skill Match", min_value=0, max_value=100, format="%d"),
                "Experience":  st.column_config.ProgressColumn("Experience",  min_value=0, max_value=100, format="%d"),
                "Trajectory":  st.column_config.ProgressColumn("Trajectory",  min_value=0, max_value=100, format="%d"),
                "Engagement":  st.column_config.ProgressColumn("Engagement",  min_value=0, max_value=100, format="%.1f"),
            }
        )

        # ── Export ───────────────────────────────────────────────────────────
        st.markdown("#### 💾 Export")
        e1,e2 = st.columns(2)
        with e1:
            st.download_button("⬇️ Download CSV", df[disp_cols].to_csv(index=False).encode(),
                file_name=f"redrob_ranked_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv", use_container_width=True)
        with e2:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w: df[disp_cols].to_excel(w, index=False, sheet_name="Ranked")
            st.download_button("⬇️ Download Excel", buf.getvalue(),
                file_name=f"redrob_ranked_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Analytics
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📊 Visual Analytics Dashboard")
    if not st.session_state.candidates:
        st.info("👆 Load data first")
    else:
        df = make_df(st.session_state.candidates, weights)
        DARK_LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_family="Inter", font_color="#f0f0f8",
                           xaxis=dict(gridcolor="#2a2a3a"), yaxis=dict(gridcolor="#2a2a3a"))

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            fig = px.histogram(df, x="Final Score", nbins=12, color="Status",
                color_discrete_map={"✅ Shortlisted":"#22c55e","👁 Review":"#f59e0b","⛔ Rejected":"#ef4444"},
                template="plotly_dark", title="Score Distribution")
            fig.update_layout(**DARK_LAYOUT, bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)

        with r1c2:
            fig = px.scatter(df, x="Exp (yrs)", y="Skill Match", size="Final Score", color="Status",
                hover_data=["Name","Company","Final Score"],
                color_discrete_map={"✅ Shortlisted":"#22c55e","👁 Review":"#f59e0b","⛔ Rejected":"#ef4444"},
                template="plotly_dark", title="Experience vs Skill Match")
            fig.add_hline(y=83, line_dash="dash", line_color="#6c63ff", annotation_text="Shortlist threshold")
            fig.update_layout(**DARK_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            top10 = df.head(10)
            cats  = ["Composite","Skill Match","Experience","Trajectory","Engagement"]
            cols  = ["#6c63ff","#a78bfa","#22c55e","#f59e0b","#ef4444","#38bdf8","#fb923c","#a3e635","#f472b6","#94a3b8"]
            fig   = go.Figure()
            for i, (_, row) in enumerate(top10.iterrows()):
                v = [row[c] for c in cats]
                fig.add_trace(go.Scatterpolar(r=v+[v[0]], theta=cats+[cats[0]],
                    fill="toself", name=row["Name"].split()[0], line_color=cols[i], opacity=0.55))
            fig.update_layout(polar=dict(
                radialaxis=dict(visible=True, range=[0,100], gridcolor="#2a2a3a", color="#8888aa"),
                angularaxis=dict(gridcolor="#2a2a3a"), bgcolor="rgba(0,0,0,0)"),
                paper_bgcolor="rgba(0,0,0,0)", font_family="Inter", font_color="#f0f0f8",
                showlegend=True, legend=dict(font=dict(size=9)), title="Top 10 Radar Comparison", title_font_size=14)
            st.plotly_chart(fig, use_container_width=True)

        with r2c2:
            fc = df["Authenticity"].value_counts()
            fig = px.pie(names=fc.index, values=fc.values, hole=0.5, template="plotly_dark",
                color=fc.index, title="Authenticity Breakdown",
                color_discrete_map={"🟢 Verified":"#22c55e","🟡 Minor Flag":"#f59e0b","🔴 Major Flag":"#ef4444"})
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_family="Inter", font_color="#f0f0f8", title_font_size=14)
            st.plotly_chart(fig, use_container_width=True)

        top15 = df.head(15)
        fig = go.Figure()
        for col, color in [("Skill Match","#6c63ff"),("Experience","#a78bfa"),("Trajectory","#22c55e"),("Engagement","#f59e0b")]:
            fig.add_trace(go.Bar(name=col, x=top15["Name"].str.split().str[0], y=top15[col], marker_color=color))
        fig.update_layout(barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_family="Inter", font_color="#f0f0f8",
            xaxis=dict(gridcolor="#2a2a3a"), yaxis=dict(gridcolor="#2a2a3a", range=[0,100]),
            legend=dict(orientation="h", y=1.1), height=360, title="Top 15 — Score Components")
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Candidate Profiles
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 🔍 Detailed Candidate Profiles")
    if not st.session_state.candidates:
        st.info("👆 Load data first")
    else:
        df = make_df(st.session_state.candidates, weights)
        selected = st.selectbox("Select candidate", df["Name"].tolist())
        if selected:
            row  = df[df["Name"] == selected].iloc[0]
            orig = next((c for c in st.session_state.candidates if c["name"] == selected), None)

            st.markdown(f"""
            <div style="background:#12121a;border:1px solid #2a2a3a;border-radius:16px;padding:24px;margin-bottom:20px;">
              <div style="display:flex;align-items:flex-start;gap:20px;flex-wrap:wrap;">
                <div style="flex:1;min-width:240px;">
                  <div style="font-size:1.4rem;font-weight:700;margin-bottom:4px;">{row['Name']}</div>
                  <div style="color:#8888aa;margin-bottom:8px;">{row['Title']} · {row['Company']}</div>
                  <div style="color:#8888aa;font-size:0.85rem;">📍 {row['Location']} · ⏱ {row['Exp (yrs)']}y exp</div>
                </div>
                <div style="text-align:right;">
                  <div style="font-size:2.5rem;font-weight:700;font-family:'JetBrains Mono',monospace;color:#6c63ff;">{row['Final Score']:.1f}</div>
                  <div style="color:#8888aa;font-size:0.75rem;">FINAL SCORE</div>
                  <div style="margin-top:8px;">{row['Status']} · {row['Authenticity']}</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

            pc1, pc2 = st.columns(2)
            with pc1:
                st.markdown("**Score Breakdown**")
                for label, val in [("Composite Fit", row["Composite"]), ("Skill Match", row["Skill Match"]),
                                    ("Experience", row["Experience"]), ("Trajectory", row["Trajectory"]),
                                    ("Engagement", row["Engagement"])]:
                    color = "#22c55e" if val >= 83 else "#f59e0b" if val >= 70 else "#ef4444"
                    st.markdown(f"""
                    <div style="margin-bottom:10px;">
                      <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                        <span style="font-size:0.8rem;color:#8888aa;">{label}</span>
                        <span style="font-size:0.8rem;font-family:'JetBrains Mono',monospace;color:{color};">{val}</span>
                      </div>
                      <div style="height:6px;background:#2a2a3a;border-radius:3px;overflow:hidden;">
                        <div style="width:{val}%;height:100%;background:linear-gradient(90deg,#6c63ff,#a78bfa);border-radius:3px;"></div>
                      </div>
                    </div>""", unsafe_allow_html=True)

            with pc2:
                st.markdown("**Skill Match vs JD Requirements**")
                strengths = orig.get("strengths", []) if orig else []
                gaps      = orig.get("gaps", []) if orig else []
                for req in ["Embeddings Retrieval","Vector Databases","Python","NDCG/MRR/MAP",
                            "End-to-End Shipping","LLM Fine-tuning","Learning-to-Rank","Open-source"]:
                    gapped  = any(req.lower()[:8] in g.lower() for g in gaps)
                    matched = any(req.lower()[:8] in s.lower() for s in strengths) or row["Skill Match"] >= 85
                    icon, color = ("❌","#ef4444") if gapped else ("✅","#22c55e") if matched else ("🔶","#f59e0b")
                    st.markdown(f'<div style="font-size:0.82rem;padding:4px 0;color:{color};">{icon} {req}</div>', unsafe_allow_html=True)

            st.divider()
            st.markdown("**🤖 AI Recruiter Reasoning**")
            reasoning = orig.get("reasoning","") if orig else ""
            st.markdown(f"""<div style="background:#1a1a26;border-left:3px solid #6c63ff;border-radius:0 10px 10px 0;
                padding:16px 20px;font-size:0.88rem;line-height:1.65;color:#cccce0;">{reasoning}</div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown("**✅ Key Strengths**")
                for s in (orig.get("strengths",[]) if orig else []):
                    st.markdown(f'<span style="display:inline-block;background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.25);color:#86efac;font-size:0.75rem;border-radius:6px;padding:4px 10px;margin:3px;">{s}</span>', unsafe_allow_html=True)
            with sc2:
                st.markdown("**⚠️ Gaps vs JD**")
                for g in (orig.get("gaps",[]) if orig else []):
                    st.markdown(f'<span style="display:inline-block;background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.25);color:#fcd34d;font-size:0.75rem;border-radius:6px;padding:4px 10px;margin:3px;">{g}</span>', unsafe_allow_html=True)

            if orig and orig.get("resume"):
                st.divider()
                with st.expander("📄 Resume Text"):
                    st.code(orig["resume"], language=None)
