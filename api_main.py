"""
╔══════════════════════════════════════════════════════════════════╗
║          LoanSense AI  —  Production FastAPI Server              ║
║  Endpoints: /predict  /explain  /health  /global-summary        ║
╚══════════════════════════════════════════════════════════════════╝

Run locally:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Run with Docker:
    docker build -t loansense-api .
    docker run -p 8000:8000 loansense-api
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# In production, send these to Datadog / CloudWatch / Grafana Loki
# ─────────────────────────────────────────────────────────────────────────────
import joblib
from sklearn.preprocessing import LabelEncoder

CATEGORICAL_FEATURES = ['home_ownership', 'loan_purpose']
NUMERIC_FEATURES = [
    'loan_amount', 'interest_rate', 'annual_income', 'dti',
    'credit_score', 'emp_length_yrs', 'num_credit_lines',
    'delinq_2yrs', 'months_since_last_delinq', 'pub_rec'
]

class LoanPreprocessor:

    def __init__(self):
        self.label_encoders = {}
        self.feature_names_out = None
        self.is_fitted = False

    def fit(self, df):
        for col in CATEGORICAL_FEATURES:
            le = LabelEncoder()
            le.fit(df[col])
            self.label_encoders[col] = le

        self.feature_names_out = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        self.is_fitted = True
        return self

    def transform(self, df):
        if not self.is_fitted:
            raise RuntimeError('Call fit() before transform()')

        result = df[NUMERIC_FEATURES].copy()

        for col in CATEGORICAL_FEATURES:
            result[col] = self.label_encoders[col].transform(df[col])

        return result

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("loansense_api")


# ─────────────────────────────────────────────────────────────────────────────
# APP INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="LoanSense AI API",
    description=(
        "Production-grade Explainable AI system for loan approval prediction. "
        "Every prediction comes with SHAP-based explanations for compliance "
        "and applicant transparency."
    ),
    version="1.0.0",
    docs_url="/docs",       # Swagger UI
    redoc_url="/redoc",     # ReDoc UI
)

# Allow the Streamlit UI (and any other frontend) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict to specific domains in real production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL STORE — Load once at startup, reuse across all requests
# Loading models on every request would be catastrophically slow
# ─────────────────────────────────────────────────────────────────────────────

class ModelStore:
    """Singleton container for all model artifacts."""

    ARTIFACTS_DIR = Path("./artifacts")

    def __init__(self):
        self.model        = None
        self.preprocessor = None
        self.explainer    = None
        self.metadata     = None
        self._loaded      = False

    def load(self):
        if self._loaded:
            return

        start = time.time()
        logger.info("Loading model artifacts from %s...", self.ARTIFACTS_DIR)

        try:
            self.model        = joblib.load(self.ARTIFACTS_DIR / "xgb_model.pkl")
            self.preprocessor = joblib.load(self.ARTIFACTS_DIR / "preprocessor.pkl")
            self.explainer    = joblib.load(self.ARTIFACTS_DIR / "shap_explainer.pkl")

            with open(self.ARTIFACTS_DIR / "model_metadata.json") as f:
                self.metadata = json.load(f)

            elapsed = time.time() - start
            logger.info("✅ All artifacts loaded in %.2fs", elapsed)
            logger.info("   Model version : %s", self.metadata["model_version"])
            logger.info("   Test AUC      : %s", self.metadata["test_auc"])
            self._loaded = True

        except FileNotFoundError as exc:
            logger.error("❌ Artifact not found: %s", exc)
            logger.error("   Run the Jupyter notebook first to generate artifacts.")
            raise


store = ModelStore()


@app.on_event("startup")
async def startup_event():
    """Load all models at startup — never during request handling."""
    store.load()


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
# Pydantic validates all inputs automatically — no manual validation needed
# ─────────────────────────────────────────────────────────────────────────────

class LoanApplication(BaseModel):
    """Incoming loan application data."""

    loan_amount:              float = Field(..., ge=500,    le=100000,  description="Requested loan amount in USD")
    interest_rate:            float = Field(..., ge=1.0,   le=36.0,    description="Proposed interest rate (%)")
    annual_income:            float = Field(..., ge=5000,  le=2000000, description="Annual income in USD")
    dti:                      float = Field(..., ge=0.0,   le=100.0,   description="Debt-to-income ratio (%)")
    credit_score:             int   = Field(..., ge=300,   le=850,     description="FICO credit score")
    emp_length_yrs:           int   = Field(..., ge=0,     le=40,      description="Years of employment")
    num_credit_lines:         int   = Field(..., ge=0,     le=100,     description="Number of open credit lines")
    delinq_2yrs:              int   = Field(..., ge=0,     le=20,      description="Delinquencies in past 2 years")
    months_since_last_delinq: int   = Field(..., ge=0,     le=240,     description="Months since last delinquency")
    pub_rec:                  int   = Field(..., ge=0,     le=20,      description="Number of public records")
    home_ownership:           str   = Field(...,                        description="RENT | MORTGAGE | OWN | OTHER")
    loan_purpose:             str   = Field(...,                        description="e.g., debt_consolidation, credit_card")

    @validator("home_ownership")
    def validate_home_ownership(cls, v):
        allowed = {"RENT", "MORTGAGE", "OWN", "OTHER"}
        if v.upper() not in allowed:
            raise ValueError(f"home_ownership must be one of: {allowed}")
        return v.upper()

    @validator("loan_purpose")
    def validate_loan_purpose(cls, v):
        allowed = {
            "debt_consolidation", "credit_card", "home_improvement",
            "other", "major_purchase", "medical", "small_business"
        }
        if v.lower() not in allowed:
            raise ValueError(f"loan_purpose must be one of: {allowed}")
        return v.lower()

    class Config:
        schema_extra = {
            "example": {
                "loan_amount": 15000,
                "interest_rate": 12.5,
                "annual_income": 75000,
                "dti": 18.5,
                "credit_score": 720,
                "emp_length_yrs": 5,
                "num_credit_lines": 12,
                "delinq_2yrs": 0,
                "months_since_last_delinq": 24,
                "pub_rec": 0,
                "home_ownership": "RENT",
                "loan_purpose": "debt_consolidation"
            }
        }


class FeatureContribution(BaseModel):
    feature:       str
    display_name:  str
    raw_value:     float
    shap_value:    float
    direction:     str  # "positive" | "negative"
    human_label:   str  # e.g., "Helps your application"


class PredictionResponse(BaseModel):
    decision:             str    # "APPROVED" | "REJECTED"
    approval_probability: float  # 0.0 – 1.0
    confidence_band:      str    # "High" | "Medium" | "Low"
    top_factors:          List[FeatureContribution]
    plain_english:        str    # Human-readable explanation
    model_version:        str
    inference_time_ms:    float


class HealthResponse(BaseModel):
    status:        str
    model_version: str
    artifacts_ok:  bool
    uptime_s:      float


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

_start_time = time.time()

FEATURE_DISPLAY = {
    "credit_score":             "Credit Score",
    "dti":                      "Debt-to-Income Ratio",
    "annual_income":            "Annual Income",
    "loan_amount":              "Loan Amount",
    "interest_rate":            "Interest Rate",
    "emp_length_yrs":           "Employment Length",
    "num_credit_lines":         "Open Credit Lines",
    "delinq_2yrs":              "Delinquencies (2yr)",
    "months_since_last_delinq": "Months Since Delinquency",
    "pub_rec":                  "Public Records",
    "home_ownership":           "Home Ownership",
    "loan_purpose":             "Loan Purpose",
}


def _confidence_band(prob: float) -> str:
    """Translate a probability into a human-friendly confidence tier."""
    margin = abs(prob - 0.5)
    if margin > 0.30:
        return "High"
    if margin > 0.15:
        return "Medium"
    return "Low"


def _build_plain_english(
    decision: str,
    prob: float,
    factors: List[FeatureContribution]
) -> str:
    """Generate a compliance-friendly, plain-English explanation."""
    pos = [f for f in factors if f.direction == "positive"][:2]
    neg = [f for f in factors if f.direction == "negative"][:2]

    if decision == "APPROVED":
        intro = (
            f"Congratulations! Your application has been approved with a "
            f"{prob:.0%} approval likelihood. "
        )
        pos_text = (
            "Your application was strengthened by: "
            + " and ".join(f.display_name for f in pos) + ". "
            if pos else ""
        )
        neg_text = (
            "Areas you may want to monitor: "
            + ", ".join(f.display_name for f in neg) + "."
            if neg else ""
        )
    else:
        intro = (
            f"Your application was not approved at this time. "
            f"Our model assigned a {prob:.0%} approval probability. "
        )
        neg_text = (
            "The primary factors affecting this decision were: "
            + " and ".join(f.display_name for f in neg) + ". "
            if neg else ""
        )
        pos_text = (
            "On the positive side: "
            + ", ".join(f.display_name for f in pos)
            + " were in your favor. "
            if pos else ""
        )
        intro += (
            "You may re-apply after addressing the factors listed below. "
            "This decision was made by an automated system and you may "
            "request a human review."
        )

    return intro + pos_text + neg_text


def _application_to_df(app: LoanApplication) -> pd.DataFrame:
    """Convert a Pydantic model to the DataFrame format the preprocessor expects."""
    return pd.DataFrame([{
        "loan_amount":              app.loan_amount,
        "interest_rate":            app.interest_rate,
        "annual_income":            app.annual_income,
        "dti":                      app.dti,
        "credit_score":             app.credit_score,
        "emp_length_yrs":           app.emp_length_yrs,
        "num_credit_lines":         app.num_credit_lines,
        "delinq_2yrs":              app.delinq_2yrs,
        "months_since_last_delinq": app.months_since_last_delinq,
        "pub_rec":                  app.pub_rec,
        "home_ownership":           app.home_ownership,
        "loan_purpose":             app.loan_purpose,
    }])


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Liveness and readiness probe — used by Kubernetes, Render, etc."""
    return HealthResponse(
        status="healthy",
        model_version=store.metadata["model_version"],
        artifacts_ok=store._loaded,
        uptime_s=round(time.time() - _start_time, 1),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(application: LoanApplication, request: Request):
    """
    Primary endpoint — predict loan approval and explain the decision.

    Returns a decision (APPROVED/REJECTED), probability, and the top
    features that drove the prediction. Designed for both applicant-facing
    UIs and internal risk dashboards.
    """
    t0 = time.perf_counter()

    try:
        # 1. Preprocess input
        raw_df = _application_to_df(application)
        X = store.preprocessor.transform(raw_df)

        # 2. Predict
        threshold = store.metadata.get("threshold", 0.5)
        prob      = float(store.model.predict_proba(X)[0, 1])
        decision  = "APPROVED" if prob >= threshold else "REJECTED"

        # 3. SHAP explanation — this is where the magic happens
        shap_vals_obj = store.explainer(X)
        shap_values = shap_vals_obj.values

        if len(shap_values.shape) == 3:
         shap_pos = shap_values[0, :, 1]   # multiclass
        else:
         shap_pos = shap_values[0]         # binary (your case) # positive class
        raw_values    = X.values[0]
        feature_names = store.preprocessor.feature_names_out

        # 4. Build factor list (sorted by |SHAP|)
        factors = []
        for feat, sv, rv in sorted(
            zip(feature_names, shap_pos, raw_values),
            key=lambda x: abs(x[1]), reverse=True
        )[:6]:
            factors.append(FeatureContribution(
                feature=feat,
                display_name=FEATURE_DISPLAY.get(feat, feat),
                raw_value=float(round(rv, 2)),
                shap_value=float(round(sv, 4)),
                direction="positive" if sv >= 0 else "negative",
                human_label="Helps your application" if sv >= 0 else "Affects your application negatively"
            ))

        # 5. Build human-readable explanation
        plain_english = _build_plain_english(decision, prob, factors)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "Prediction: %s | prob=%.3f | time=%.1fms",
            decision, prob, elapsed_ms
        )

        return PredictionResponse(
            decision=decision,
            approval_probability=round(prob, 4),
            confidence_band=_confidence_band(prob),
            top_factors=factors,
            plain_english=plain_english,
            model_version=store.metadata["model_version"],
            inference_time_ms=round(elapsed_ms, 2),
        )

    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(exc)}")


@app.get("/global-summary", tags=["Explainability"])
async def global_summary():
    """
    Return global model statistics and feature importance.

    Used by compliance teams and product dashboards to understand
    overall model behavior — not specific to any single applicant.
    """
    metadata = store.metadata
    return {
        "model_version":       metadata["model_version"],
        "trained_at":          metadata["trained_at"],
        "test_roc_auc":        metadata["test_auc"],
        "test_f1":             metadata["test_f1"],
        "n_training_samples":  metadata["n_train_samples"],
        "decision_threshold":  metadata["threshold"],
        "baseline_probability":metadata["baseline_value"],
        "features":            metadata["features"],
        "description": (
            "This model uses XGBoost with SHAP explanations. "
            "It was trained on historical loan data and validated "
            "with 5-fold cross-validation. Every prediction includes "
            "feature-level explanations for regulatory compliance."
        )
    }


@app.get("/", tags=["System"])
async def root():
    return {
        "service": "LoanSense AI",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/health",
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
