"""Feature engineering for the scam-risk model (payer, session, payee, velocity)."""
import numpy as np
import pandas as pd

from . import config as C

# Signal groups map to data sources a bank would have (or could obtain):
#   payer profile      -> core banking / CRM
#   session / device   -> mobile banking SDK, device binding, telco APIs
#   payee intelligence -> MuleHunter.AI score, CFCFRMS (I4C 1930) reports, DPIP
#   velocity           -> real-time feature store
FEATURES = [
    # amount & payer
    "log_amount", "amount_to_median_ratio", "balance_drain_ratio", "age", "is_vulnerable",
    "tenure_months", "digital_literacy", "city_tier",
    # session / device / telco
    "on_active_call", "call_duration_min", "caller_unknown", "caller_voip_or_intl", "video_call",
    "screen_share_active", "remote_access_app", "new_device_7d", "sim_swap_7d", "odd_hour",
    "remark_risk_kw",
    # payee intelligence
    "payee_is_new", "log_payee_account_age", "payee_fan_in_24h", "payee_passthrough_ratio",
    "payee_mule_score", "payee_reported_cfcfrms",
    # velocity
    "payer_txn_count_1h", "payer_txn_count_24h", "log_payer_amt_24h", "log_payer_new_payee_amt_7d",
]

REASON_TEXT = {
    "log_amount": "Unusually large amount",
    "amount_to_median_ratio": "Amount far above customer's normal",
    "balance_drain_ratio": "Payment drains most of the balance",
    "age": "Customer age profile is high-risk for this pattern",
    "is_vulnerable": "Senior citizen / person with disability",
    "on_active_call": "Customer is on a live call while paying",
    "call_duration_min": "Very long ongoing call (coaching pattern)",
    "caller_unknown": "Caller not in contacts",
    "caller_voip_or_intl": "VoIP / international caller",
    "video_call": "Ongoing video call (digital-arrest pattern)",
    "screen_share_active": "Screen sharing is active",
    "remote_access_app": "Remote-access app detected",
    "new_device_7d": "New device in last 7 days",
    "sim_swap_7d": "Recent SIM swap",
    "odd_hour": "Unusual time of day",
    "remark_risk_kw": "Payment remark matches scam keywords",
    "payee_is_new": "First payment to this beneficiary",
    "log_payee_account_age": "Beneficiary account is newly opened",
    "payee_fan_in_24h": "Beneficiary receiving from many payers",
    "payee_passthrough_ratio": "Beneficiary moves money out quickly (pass-through)",
    "payee_mule_score": "High mule-account score for beneficiary",
    "payee_reported_cfcfrms": "Beneficiary reported on 1930 / CFCFRMS",
    "payer_txn_count_1h": "Burst of payments in last hour",
    "payer_txn_count_24h": "Many payments in last 24h",
    "log_payer_amt_24h": "High total outflow in last 24h",
    "log_payer_new_payee_amt_7d": "Rising payments to new beneficiaries this week",
}


def build(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["log_amount"] = np.log1p(df.amount)
    X["amount_to_median_ratio"] = df.amount_to_median_ratio.clip(upper=500)
    X["balance_drain_ratio"] = df.balance_drain_ratio
    X["age"] = df.age
    X["is_vulnerable"] = ((df.age >= C.SENIOR_AGE) | df.is_disabled).astype(int)
    X["tenure_months"] = df.tenure_months
    X["digital_literacy"] = df.digital_literacy
    X["city_tier"] = df.city_tier
    for b in ["on_active_call", "caller_unknown", "caller_voip_or_intl", "video_call",
              "screen_share_active", "remote_access_app", "new_device_7d", "sim_swap_7d",
              "remark_risk_kw", "payee_is_new", "payee_reported_cfcfrms"]:
        X[b] = df[b].astype(int)
    X["call_duration_min"] = df.call_duration_min
    X["odd_hour"] = ((df.hour < 7) | (df.hour >= 22)).astype(int)
    X["log_payee_account_age"] = np.log1p(df.payee_account_age_days)
    X["payee_fan_in_24h"] = df.payee_fan_in_24h
    X["payee_passthrough_ratio"] = df.payee_passthrough_ratio
    X["payee_mule_score"] = df.payee_mule_score
    X["payer_txn_count_1h"] = df.payer_txn_count_1h
    X["payer_txn_count_24h"] = df.payer_txn_count_24h
    X["log_payer_amt_24h"] = np.log1p(df.payer_amt_24h.clip(lower=0))
    X["log_payer_new_payee_amt_7d"] = np.log1p(df.payer_new_payee_amt_7d.clip(lower=0))
    return X[FEATURES]
