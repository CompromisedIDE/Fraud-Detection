import streamlit as st
import requests
import json

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="shield",
    layout="centered"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: #0d1117;
        color: #c9d1d9;
    }

    .block-container { padding-top: 48px; max-width: 720px; }

    h1 {
        font-size: 24px;
        font-weight: 700;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #f0f6fc;
        margin-bottom: 4px;
    }

    .subtitle {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        color: #6e7681;
        letter-spacing: 1px;
        margin-bottom: 32px;
    }

    .divider {
        border: none;
        border-top: 1px solid #21262d;
        margin: 24px 0;
    }

    .section-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px;
        color: #6e7681;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .stTextArea textarea {
        background-color: #161b22 !important;
        border: 1px solid #21262d !important;
        border-radius: 4px !important;
        color: #c9d1d9 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 12px !important;
    }

    .stTextArea textarea:focus {
        border-color: #e63946 !important;
        box-shadow: 0 0 0 3px rgba(230,57,70,0.08) !important;
    }

    .stButton > button {
        background-color: #e63946 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 4px !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 13px !important;
        font-weight: 700 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        padding: 10px 28px !important;
        width: 100% !important;
        margin-top: 8px !important;
    }

    .stButton > button:hover {
        background-color: #c1121f !important;
    }

    .result-fraud {
        background: rgba(230,57,70,0.07);
        border: 1px solid rgba(230,57,70,0.3);
        border-radius: 4px;
        padding: 20px 24px;
        margin-top: 24px;
    }

    .result-legit {
        background: rgba(63,185,80,0.07);
        border: 1px solid rgba(63,185,80,0.3);
        border-radius: 4px;
        padding: 20px 24px;
        margin-top: 24px;
    }

    .result-status-fraud {
        font-size: 18px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #e63946;
        margin-bottom: 6px;
    }

    .result-status-legit {
        font-size: 18px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #3fb950;
        margin-bottom: 6px;
    }

    .result-prob {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        color: #8b949e;
    }

    .api-status-ok {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        color: #3fb950;
    }

    .api-status-err {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        color: #e63946;
    }

    div[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #21262d;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="section-label">API Status</div>', unsafe_allow_html=True)
    try:
        requests.get("http://127.0.0.1:8000/health", timeout=3)
        st.markdown('<div class="api-status-ok">● FastAPI connected</div>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<div class="api-status-err">● FastAPI offline</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Model</div>', unsafe_allow_html=True)
    st.caption("XGBoost classifier. Class imbalance handled with SMOTE. Features V1-V28 are PCA-transformed components from the Kaggle Credit Card Fraud dataset.")

# Header
st.markdown("<h1>Fraud Detection</h1>", unsafe_allow_html=True)
st.markdown('<div class="subtitle">// XGBoost · SMOTE · FastAPI · Real-time</div>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# Default payload
default_payload = {
    "Time": 0.0, "Amount": 149.62,
    "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155,
    "V5": -0.338321, "V6": 0.462388, "V7": 0.239599, "V8": 0.098698,
    "V9": 0.363787, "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
    "V13": -0.991390, "V14": -0.311169, "V15": 1.468177, "V16": -0.470401,
    "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
    "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
    "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053
}

# Input
st.markdown('<div class="section-label">Transaction Payload</div>', unsafe_allow_html=True)
user_input = st.text_area(
    label="Transaction Payload",
    value=json.dumps(default_payload, indent=2),
    height=260,
    label_visibility="collapsed"
)

# Predict
if st.button("Analyze Transaction"):
    try:
        payload = json.loads(user_input)

        with st.spinner("Running analysis..."):
            response = requests.post(API_URL, json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()
            is_fraud = result["is_fraud"]
            prob = result["fraud_probability_percent"]
            status = result["status"]

            box_class = "result-fraud" if is_fraud else "result-legit"
            status_class = "result-status-fraud" if is_fraud else "result-status-legit"

            st.markdown(f"""
            <div class="{box_class}">
                <div class="{status_class}">{status}</div>
                <div class="result-prob">Fraud probability: {prob}%</div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.error(f"API error {response.status_code}: {response.text}")

    except json.JSONDecodeError:
        st.error("Invalid JSON. Check the payload format and try again.")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to FastAPI. Make sure the server is running on port 8000.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
