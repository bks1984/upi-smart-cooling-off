"""
Real-time decision API (reference implementation).

    uvicorn scooler.service:app --app-dir src --reload
    POST /v1/decide   -> action, delay, risk score, reason codes, customer message

In production this sits in the payer bank's payment-initiation path (before the
UPI / IMPS request is released to NPCI). Velocity features come from a real-time
feature store; payee intelligence from MuleHunter.AI / DPIP / CFCFRMS lookups.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import config as C
from .features import build
from .model import RiskModel
from .policy import Thresholds, decide_batch, warning_for

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models/risk_model.joblib"
RESULTS_PATH = ROOT / "reports/results.json"


class PaymentRequest(BaseModel):
    amount: float = Field(..., gt=0, examples=[185000])
    # payer
    age: int = Field(..., examples=[72])
    is_disabled: bool = False
    tenure_months: int = 60
    digital_literacy: float = 0.5
    city_tier: int = 2
    median_amt: float = Field(2500, description="Customer's median P2P amount (90d)")
    balance_before: float = Field(..., examples=[240000])
    # session / device / telco
    on_active_call: bool = False
    call_duration_min: float = 0
    caller_unknown: bool = False
    caller_voip_or_intl: bool = False
    video_call: bool = False
    screen_share_active: bool = False
    remote_access_app: bool = False
    new_device_7d: bool = False
    sim_swap_7d: bool = False
    hour: float = 14
    remark_risk_kw: bool = False
    # payee intelligence
    payee_is_new: bool = True
    payee_account_age_days: float = 1000
    payee_fan_in_24h: int = 0
    payee_passthrough_ratio: float = 0.2
    payee_mule_score: float = 0.05
    payee_reported_cfcfrms: bool = False
    # velocity (from feature store)
    payer_txn_count_1h: int = 0
    payer_txn_count_24h: int = 0
    payer_amt_24h: float = 0
    payer_new_payee_amt_7d: float = 0


class Decision(BaseModel):
    action: str
    delay_minutes: int
    p_scam: float
    expected_loss_inr: float
    guardrail: str
    reasons: list[str]
    customer_message: str


CUSTOMER_COPY = {
    "allow": "Payment sent.",
    "nudge": "Please check before you pay. {warning}",
    "cool_60": "For your safety this payment will be sent in 60 minutes. You can cancel any time "
               "before then from the app. {warning}",
    "hold_callback": "This payment is on hold. Your bank (or your registered trusted person) will "
                     "call you to confirm before it is released. {warning}",
    "block": "This payment was stopped because it matches a known scam pattern. {warning} "
             "If you have already lost money, call 1930 immediately.",
    "block_cfcfrms": "This payment was stopped because the receiver's account has been reported for "
                     "fraud on the National Cyber Crime Portal. If you have lost money, call 1930 immediately.",
}


def _load():
    model = RiskModel.load(str(MODEL_PATH))
    th = Thresholds(**json.loads(RESULTS_PATH.read_text())["thresholds"])
    return model, th


app = FastAPI(title="Smart Cooling-Off - APP scam intervention engine", version="0.1.0")
_state: dict = {}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/decide", response_model=Decision)
def decide(req: PaymentRequest) -> Decision:
    if "model" not in _state:
        _state["model"], _state["th"] = _load()
    model, th = _state["model"], _state["th"]
    row = req.model_dump()
    row["amount_to_median_ratio"] = row["amount"] / max(row["median_amt"], 1)
    row["balance_drain_ratio"] = min(row["amount"] / max(row["balance_before"], 1), 1.5)
    df = pd.DataFrame([row])
    X = build(df)
    p = model.score(X)
    d = decide_batch(df, p, th).iloc[0]
    warn = warning_for(df.iloc[0])
    return Decision(
        action=d.action, delay_minutes=C.DELAY_MINUTES[d.action], p_scam=round(float(p[0]), 4),
        expected_loss_inr=round(float(p[0] * req.amount), 2), guardrail=d.rule,
        reasons=model.reasons(X)[0] if d.action != "allow" else [],
        customer_message=CUSTOMER_COPY["block_cfcfrms" if d.rule == "cfcfrms_mule_hit" else d.action]
        .format(warning=warn),
    )
