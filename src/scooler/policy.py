"""
Smart Cooling-Off decision engine.

Turns a calibrated scam probability into a graduated intervention, instead of a
flat timer on every payment above a threshold:

  allow          -> instant, as today
  nudge          -> typology-specific scam warning + re-confirm (no delay)
  cool_60        -> 60-min cooling-off: provisional debit, cancel button, SMS/push
                    alert, payee re-screened by mule / DPIP intelligence
  hold_callback  -> hold up to 4h; outbound bank call-back or "trusted person"
                    approval (RBI Option 2) before release
  block          -> decline; customer referred to 1930 / branch

Decisions are driven by expected loss (p_scam x amount) so that friction is spent
where money is actually at risk, plus a small set of policy guard-rails that
encode regulatory intent and known scam mechanics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import config as C

ORDER = {a: i for i, a in enumerate(C.ACTIONS)}


@dataclass
class Thresholds:
    el_nudge: float = 300.0       # expected loss (Rs) to show a warning
    el_cool: float = 1_500.0      # expected loss to impose a 60-min cooling-off
    el_hold: float = 15_000.0     # expected loss to require call-back / trusted person
    p_block: float = 0.90         # probability above which payment is declined
    trusted_payee_p: float = 0.05  # repeat payees below this risk are never delayed

    def to_dict(self):
        return asdict(self)


def _max_action(a: np.ndarray, floor: str, mask: np.ndarray) -> np.ndarray:
    lvl = np.vectorize(ORDER.get)(a)
    lvl = np.where(mask, np.maximum(lvl, ORDER[floor]), lvl)
    return np.array(C.ACTIONS, dtype=object)[lvl]


def decide_batch(df: pd.DataFrame, p: np.ndarray, th: Thresholds) -> pd.DataFrame:
    """Vectorised decisions for a batch of APP transactions.

    Returns DataFrame with columns: action, rule (which guard-rail fired, if any).
    """
    el = p * df.amount.values
    action = np.full(len(df), "allow", dtype=object)
    action[el >= th.el_nudge] = "nudge"
    action[el >= th.el_cool] = "cool_60"
    action[el >= th.el_hold] = "hold_callback"
    action[p >= th.p_block] = "block"
    rule = np.full(len(df), "risk_score", dtype=object)

    # G1 trusted repeat payee: never add friction when risk is genuinely low
    trusted = (~df.payee_is_new.values) & (p < th.trusted_payee_p)
    action[trusted] = "allow"
    rule[trusted] = "trusted_payee"

    # G2 coached-victim pattern: long call from unknown number while paying a new
    # payee >= Rs 10k. A passive timer does not help (the fraudster stays on the
    # line), so escalate to an active bank call-back.
    coached = (df.on_active_call.values & df.caller_unknown.values & (df.call_duration_min.values > 30)
               & df.payee_is_new.values & (df.amount.values >= C.RBI_LAG_THRESHOLD) & (p >= 0.02))
    action = _max_action(action, "hold_callback", coached)
    rule[coached] = "coached_victim_callback"

    # G3 RBI Option 2: seniors (70+) and persons with disabilities, new payee
    # >= Rs 50k -> trusted-person / call-back approval (risk-gated to limit friction)
    vuln = (((df.age.values >= C.SENIOR_AGE) | df.is_disabled.values) & df.payee_is_new.values
            & (df.amount.values >= C.TRUSTED_PERSON_THRESHOLD) & (p >= 0.01))
    action = _max_action(action, "hold_callback", vuln)
    rule[vuln & (rule == "risk_score")] = "rbi_opt2_vulnerable"

    # G4 hard intelligence hit: beneficiary reported on 1930/CFCFRMS + high mule score
    hard = df.payee_reported_cfcfrms.values & (df.payee_mule_score.values > 0.6)
    action[hard] = "block"
    rule[hard] = "cfcfrms_mule_hit"
    rule[(action == "block") & ~hard] = "high_risk_block"

    return pd.DataFrame({"action": action, "rule": rule}, index=df.index)


WARNING_TEXT = {
    "investment": "Guaranteed-return trading / IPO apps that ask you to pay individuals are a top scam in India.",
    "digital_arrest": "Police, CBI, RBI or customs NEVER ask you to transfer money or stay on a video call. "
                      "There is no such thing as a 'digital arrest'. Hang up and call 1930.",
    "job_task": "Online 'tasks' that require you to pay to unlock earnings are scams.",
    "kyc_impersonation": "Your bank will never ask you to share your screen or install an app to update KYC.",
    "sextortion": "If someone is threatening you after a video call, do not pay. Report at cybercrime.gov.in.",
    "generic": "This payment looks unusual. Only continue if you personally know the receiver.",
}


def warning_for(row: pd.Series) -> str:
    """Pick a contextual warning from session signals (typology is unknown at run time)."""
    if row.video_call and row.call_duration_min > 30:
        return WARNING_TEXT["digital_arrest"]
    if row.screen_share_active or row.remote_access_app:
        return WARNING_TEXT["kyc_impersonation"]
    if row.remark_risk_kw and row.amount >= 10_000:
        return WARNING_TEXT["investment"]
    if row.payer_txn_count_24h >= 3 and row.payee_is_new:
        return WARNING_TEXT["job_task"]
    return WARNING_TEXT["generic"]
