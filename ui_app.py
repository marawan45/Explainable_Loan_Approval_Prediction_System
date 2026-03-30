"""
╔══════════════════════════════════════════════════════════════════╗
║         LoanSense AI  —  Streamlit Frontend                      ║
║  A production-quality UI that calls the FastAPI backend          ║
╚══════════════════════════════════════════════════════════════════╝

Run locally:
    streamlit run app.py

The app communicates with the FastAPI backend at API_URL.
Set the env variable API_URL to override the default.
"""

import json
import os
import time

import plotly.graph_objects as go
import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="LoanSense AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "LoanSense AI — Explainable Loan Approval System v1.0"},
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM STYLING
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .hero-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.8rem;
        color: #0f2027;
        line-height: 1.2;
    }

    .hero-sub {
        font-size: 1.1rem;
        color: #5a6a75;
        margin-top: 0.4rem;
    }

    .decision-approved {
        background: linear-gradient(135deg, #d4efdf, #a9dfbf);
        border-left: 5px solid #27ae60;
        border-radius: 12px;
        padding: 24px 28px;
        font-size: 1.4rem;
        font-weight: 600;
        color: #1a6b35;
    }

    .decision-rejected {
        background: linear-gradient(135deg, #fadbd8, #f5b7b1);
        border-left: 5px solid #e74c3c;
        border-radius: 12px;
        padding: 24px 28px;
        font-size: 1.4rem;
        font-weight: 600;
        color: #922b21;
    }

    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f2027;
    }

    .metric-label {
        font-size: 0.82rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .section-header {
        font-family: 'DM Serif Display', serif;
        font-size: 1.3rem;
        color: #0f2027;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }

    .plain-english-box {
        background: #eaf4fb;
        border-left: 4px solid #3498db;
        border-radius: 8px;
        padding: 16px 20px;
        font-size: 0.98rem;
        color: #1a3c5e;
        line-height: 1.7;
    }

    .factor-positive {
        background: #eafaf1;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 6px;
        border-left: 3px solid #27ae60;
        font-size: 0.9rem;
    }

    .factor-negative {
        background: #fdedec;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 6px;
        border-left: 3px solid #e74c3c;
        font-size: 0.9rem;
    }

    .stButton > button {
        background: linear-gradient(135deg, #0f2027, #203a43);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 14px 32px;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        letter-spacing: 0.03em;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(15,32,39,0.35);
    }

    .api-tag {
        font-size: 0.7rem;
        background: #e8f4f8;
        color: #2980b9;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def call_predict_api(payload: dict) -> dict:
    """Call the FastAPI /predict endpoint."""
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
        r.raise_for_status()
        return {"ok": True, "data": r.json()}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "error": "Cannot reach API server. Is it running?"}
    except requests.exceptions.Timeout:
        return {"ok": False, "error": "API request timed out."}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def call_health_api() -> dict:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.json()
    except Exception:
        return {"status": "unreachable"}


def build_shap_waterfall(factors: list, decision: str, prob: float) -> go.Figure:
    """Build a beautiful waterfall chart from SHAP factors."""
    names  = [f["display_name"] for f in factors]
    values = [f["shap_value"]   for f in factors]
    colors = ["#27ae60" if f["direction"] == "positive" else "#e74c3c" for f in factors]
    raw    = [f["raw_value"] for f in factors]

    # Build waterfall
    fig = go.Figure(go.Waterfall(
        orientation="h",
        measure=["relative"] * len(factors),
        x=values,
        y=[f"{n}<br><span style='color:#999;font-size:10px'>(value: {r})</span>"
           for n, r in zip(names, raw)],
        connector={"line": {"color": "#dee2e6", "width": 1}},
        increasing={"marker": {"color": "#27ae60"}},
        decreasing={"marker": {"color": "#e74c3c"}},
        text=[f"{'+' if v >= 0 else ''}{v:.3f}" for v in values],
        textposition="outside",
        textfont={"size": 11, "color": "#2c3e50"},
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>Why this decision was made</b><br>"
                 f"<span style='font-size:13px;color:gray'>"
                 f"Approval probability: {prob:.1%} → {'✅ APPROVED' if decision=='APPROVED' else '❌ REJECTED'}"
                 f"</span>",
            font=dict(size=16),
        ),
        xaxis_title="SHAP Value (contribution to approval probability)",
        height=max(350, len(factors) * 55 + 100),
        margin=dict(l=220, r=80, t=90, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            gridcolor="#f0f0f0",
            zeroline=True,
            zerolinecolor="#aaa",
            zerolinewidth=1.5,
            range=[-0.5, 0.5],
        ),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )

    return fig


def build_gauge(prob: float) -> go.Figure:
    """Approval probability gauge chart."""
    color = "#27ae60" if prob >= 0.5 else "#e74c3c"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(prob * 100, 1),
        delta={"reference": 50, "valueformat": ".1f", "suffix": "%",
               "font": {"size": 14}},
        number={"suffix": "%", "font": {"size": 32, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1,
                     "tickcolor": "#aaa", "ticksuffix": "%"},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  40],  "color": "#fde8e6"},
                {"range": [40, 60],  "color": "#fef9e7"},
                {"range": [60, 100], "color": "#eafaf1"},
            ],
            "threshold": {
                "line": {"color": "#2c3e50", "width": 3},
                "thickness": 0.75,
                "value": 50,
            },
        },
        title={"text": "Approval Probability", "font": {"size": 14, "color": "#555"}},
    ))

    fig.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="white",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — API Status + About
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🏦 LoanSense AI")
    st.markdown("*Explainable Loan Approval System*")
    st.divider()

    # API health check
    health = call_health_api()
    status_color = "🟢" if health.get("status") == "healthy" else "🔴"
    st.markdown(f"**API Status** {status_color} `{health.get('status', 'unknown')}`")
    if health.get("model_version"):
        st.caption(f"Model v{health['model_version']} · Uptime {health.get('uptime_s', '?')}s")

    st.divider()
    st.markdown("**About**")
    st.markdown("""
This system uses **XGBoost + SHAP** to:
- Predict loan approval probability
- Explain *exactly why* using feature attributions
- Provide human-readable justifications

Built for demonstrating production-grade **Explainable AI**.
    """)
    st.divider()
    st.markdown("**Tech Stack**")
    st.markdown("`XGBoost` `SHAP` `FastAPI` `Streamlit` `Plotly`")
    st.divider()
    st.caption("© LoanSense AI — Portfolio Demo")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

# Hero header
st.markdown("""
<div style='padding: 10px 0 24px 0;'>
    <div class='hero-title'>🏦 LoanSense AI</div>
    <div class='hero-sub'>
        Submit a loan application and receive an AI-powered decision
        with a clear, auditable explanation — instantly.
    </div>
</div>
""", unsafe_allow_html=True)

# ── TWO COLUMN LAYOUT ────────────────────────────────────────────
col_form, col_results = st.columns([1, 1.3], gap="large")

# ─────────────────────────────────────────────────────────────────────────────
# LEFT COLUMN — Application Form
# ─────────────────────────────────────────────────────────────────────────────

with col_form:
    st.markdown('<div class="section-header">📋 Loan Application</div>', unsafe_allow_html=True)

    with st.container():
        # Financial Information
        st.markdown("**💰 Financial Information**")
        c1, c2 = st.columns(2)
        loan_amount   = c1.number_input("Loan Amount ($)",       min_value=500,    max_value=100000, value=15000, step=500)
        annual_income = c2.number_input("Annual Income ($)",     min_value=10000,  max_value=500000, value=75000, step=1000)
        interest_rate = c1.slider("Interest Rate (%)",           min_value=1.0,    max_value=36.0,   value=12.5,  step=0.5)
        dti           = c2.slider("Debt-to-Income Ratio (%)",    min_value=0.0,    max_value=60.0,   value=18.5,  step=0.5)

        st.markdown("**📊 Credit Profile**")
        c3, c4 = st.columns(2)
        credit_score      = c3.slider("Credit Score",            min_value=300,    max_value=850,    value=720)
        emp_length        = c4.slider("Employment Length (yrs)", min_value=0,      max_value=40,     value=5)
        num_credit_lines  = c3.number_input("Open Credit Lines", min_value=0,      max_value=50,     value=12)
        delinq_2yrs       = c4.number_input("Delinquencies (2yr)",min_value=0,     max_value=10,     value=0)
        months_delinq     = c3.number_input("Months Since Last Delinquency", min_value=0, max_value=240, value=24)
        pub_rec           = c4.number_input("Public Records",    min_value=0,      max_value=10,     value=0)

        st.markdown("**🏠 Personal Details**")
        c5, c6 = st.columns(2)
        home_ownership = c5.selectbox(
            "Home Ownership",
            ["RENT", "MORTGAGE", "OWN", "OTHER"],
            index=0
        )
        loan_purpose = c6.selectbox(
            "Loan Purpose",
            ["debt_consolidation", "credit_card", "home_improvement",
             "other", "major_purchase", "medical", "small_business"],
            index=0
        )

    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.button("🔍 Analyze Application", type="primary")

    # Quick load example applicants
    st.divider()
    st.markdown("**🧪 Load Example Applicants**")
    eg_col1, eg_col2, eg_col3 = st.columns(3)

    # These are handled via session_state in a production app;
    # here we label them for UI clarity
    eg_col1.button("✅ Strong Profile",  key="eg1", help="720 credit, low DTI")
    eg_col2.button("⚠️  Borderline",     key="eg2", help="620 credit, mid DTI")
    eg_col3.button("❌ Weak Profile",    key="eg3", help="540 credit, high DTI")


# ─────────────────────────────────────────────────────────────────────────────
# RIGHT COLUMN — Results
# ─────────────────────────────────────────────────────────────────────────────

with col_results:
    st.markdown('<div class="section-header">📊 AI Decision & Explanation</div>', unsafe_allow_html=True)

    # Handle example buttons (set default values in session_state)
    presets = {
        "eg1": {"credit_score": 750, "dti": 12.0, "annual_income": 95000,
                "delinq_2yrs": 0, "loan_amount": 10000, "interest_rate": 9.5,
                "emp_length_yrs": 8, "pub_rec": 0},
        "eg2": {"credit_score": 625, "dti": 28.0, "annual_income": 52000,
                "delinq_2yrs": 1, "loan_amount": 18000, "interest_rate": 16.5,
                "emp_length_yrs": 3, "pub_rec": 0},
        "eg3": {"credit_score": 540, "dti": 42.0, "annual_income": 31000,
                "delinq_2yrs": 3, "loan_amount": 25000, "interest_rate": 24.0,
                "emp_length_yrs": 1, "pub_rec": 1},
    }
    active_preset = None
    for k in presets:
        if st.session_state.get(k):
            active_preset = presets[k]

    if active_preset:
        # Override form values from preset
        payload = {
            "loan_amount":              active_preset.get("loan_amount", loan_amount),
            "interest_rate":            active_preset.get("interest_rate", interest_rate),
            "annual_income":            active_preset.get("annual_income", annual_income),
            "dti":                      active_preset.get("dti", dti),
            "credit_score":             active_preset.get("credit_score", credit_score),
            "emp_length_yrs":           active_preset.get("emp_length_yrs", emp_length),
            "num_credit_lines":         num_credit_lines,
            "delinq_2yrs":              active_preset.get("delinq_2yrs", delinq_2yrs),
            "months_since_last_delinq": months_delinq,
            "pub_rec":                  active_preset.get("pub_rec", pub_rec),
            "home_ownership":           home_ownership,
            "loan_purpose":             loan_purpose,
        }
        result = call_predict_api(payload)
    elif submit:
        payload = {
            "loan_amount":              float(loan_amount),
            "interest_rate":            float(interest_rate),
            "annual_income":            float(annual_income),
            "dti":                      float(dti),
            "credit_score":             int(credit_score),
            "emp_length_yrs":           int(emp_length),
            "num_credit_lines":         int(num_credit_lines),
            "delinq_2yrs":              int(delinq_2yrs),
            "months_since_last_delinq": int(months_delinq),
            "pub_rec":                  int(pub_rec),
            "home_ownership":           home_ownership,
            "loan_purpose":             loan_purpose,
        }
        with st.spinner("Analyzing application..."):
            result = call_predict_api(payload)
    else:
        result = None

    # ── RENDER RESULTS ──────────────────────────────────────────
    if result is None:
        st.markdown("""
        <div style='padding: 40px 20px; text-align: center; color: #aaa;
                    border: 2px dashed #dee2e6; border-radius: 12px;'>
            <div style='font-size: 3rem;'>🏦</div>
            <div style='font-size: 1.1rem; margin-top: 12px;'>
                Fill in the application form and click<br>
                <strong>Analyze Application</strong> to see the AI decision.
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif not result["ok"]:
        st.error(f"⚠️ API Error: {result['error']}")
        st.info("Make sure the FastAPI server is running: `uvicorn main:app --reload`")

    else:
        data = result["data"]
        decision = data["decision"]
        prob     = data["approval_probability"]
        factors  = data["top_factors"]
        band     = data["confidence_band"]

        # Decision banner
        if decision == "APPROVED":
            st.markdown(
                f"<div class='decision-approved'>✅ APPROVED &nbsp;·&nbsp; "
                f"{prob:.1%} Approval Probability &nbsp;·&nbsp; Confidence: {band}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='decision-rejected'>❌ REJECTED &nbsp;·&nbsp; "
                f"{prob:.1%} Approval Probability &nbsp;·&nbsp; Confidence: {band}</div>",
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Quick metrics row
        m1, m2, m3 = st.columns(3)
        m1.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{prob:.1%}</div>
                <div class='metric-label'>Approval Score</div>
            </div>
        """, unsafe_allow_html=True)
        m2.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{band}</div>
                <div class='metric-label'>Confidence</div>
            </div>
        """, unsafe_allow_html=True)
        m3.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{data['inference_time_ms']:.0f}ms</div>
                <div class='metric-label'>Inference Time</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Gauge chart + SHAP waterfall
        g1, g2 = st.columns([1, 2])
        with g1:
            st.plotly_chart(build_gauge(prob), use_container_width=True, config={"displayModeBar": False})
        with g2:
            # Top factors summary
            st.markdown("**Key Decision Factors**")
            for f in factors[:5]:
                if f["direction"] == "positive":
                    st.markdown(
                        f"<div class='factor-positive'>✅ <strong>{f['display_name']}</strong>"
                        f" = {f['raw_value']} &nbsp;·&nbsp; {f['human_label']}"
                        f" <code>+{f['shap_value']:.3f}</code></div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div class='factor-negative'>⚠️ <strong>{f['display_name']}</strong>"
                        f" = {f['raw_value']} &nbsp;·&nbsp; {f['human_label']}"
                        f" <code>{f['shap_value']:.3f}</code></div>",
                        unsafe_allow_html=True
                    )

        # Full SHAP waterfall
        st.markdown("**📊 SHAP Explanation Waterfall**")
        fig_waterfall = build_shap_waterfall(factors, decision, prob)
        st.plotly_chart(fig_waterfall, use_container_width=True, config={"displayModeBar": False})

        # Plain English explanation
        st.markdown("**💬 Plain-English Explanation**")
        st.markdown(
            f"<div class='plain-english-box'>{data['plain_english']}</div>",
            unsafe_allow_html=True
        )

        # Expandable: raw API response (useful for developers)
        with st.expander("🔧 View Raw API Response (JSON)"):
            st.json(data)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown("""
<div style='text-align:center; color: #aaa; font-size:0.82rem; padding: 8px 0;'>
    LoanSense AI &nbsp;·&nbsp; Portfolio Demo &nbsp;·&nbsp;
    Built with XGBoost · SHAP · FastAPI · Streamlit &nbsp;·&nbsp;
    <em>This is a demonstration system. Not for use in real lending decisions.</em>
</div>
""", unsafe_allow_html=True)
