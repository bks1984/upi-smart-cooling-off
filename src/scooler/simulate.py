"""
Counterfactual policy simulator.

Compares, on the same out-of-time test window:
  P0  status quo (instant payments, no intervention)
  P1  RBI Option 1: flat 60-min lag on APP >= Rs 10,000 with whitelisting
  P2  Banks' ask: flat 60-min lag on APP >= Rs 25,000 with whitelisting
  P3  Smart Cooling-Off: risk-based graduated interventions (this project)

Scam losses use an *expected-value* incident model: within an incident, a
stopped payment may alert the victim and break the chain of later payments.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import config as C
from .policy import Thresholds, decide_batch

EFF = C.Effectiveness()


def flat_lag_actions(df: pd.DataFrame, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns (action, whitelist_prob) for a flat-lag policy."""
    in_scope = df.amount.values >= threshold
    action = np.where(in_scope, "cool_60", "allow").astype(object)
    wl = np.where(df.is_scam.values,
                  df.typology.map(C.COACHED_WHITELIST_PROB).fillna(0).values,
                  np.where(df.payee_is_new.values, 0.20, C.LEGIT_WHITELIST_SHARE))
    return action, wl


def _p_stop(df: pd.DataFrame, action: np.ndarray, wl: np.ndarray | None) -> np.ndarray:
    base = np.zeros(len(df))
    sc = np.flatnonzero(df.is_scam.values)
    base[sc] = [EFF.p_stop(t, a) for t, a in zip(df.typology.values[sc], action[sc])]
    hours = np.array([C.DELAY_MINUTES[a] / 60 for a in action])
    boost = np.clip(EFF.payee_intel_boost_per_hour * hours * df.payee_mule_score.values, 0, 0.5)
    p = 1 - (1 - base) * (1 - np.where(hours > 0, boost, 0))
    p = np.where(action == "block", np.maximum(p, base), p)
    if wl is not None:
        p = p * (1 - wl)           # whitelisted payments bypass the lag entirely
    return np.clip(p, 0, 1)


def evaluate_policy(df: pd.DataFrame, action: np.ndarray, wl: np.ndarray | None = None) -> dict:
    df = df.assign(action=action)
    eff_wl = np.zeros(len(df)) if wl is None else wl
    legit = ~df.is_scam.values

    # ---- friction on legitimate customers
    fric_unit = np.array([C.FRICTION_COST[a] for a in action]) * (1 - eff_wl)
    friction = float(fric_unit[legit].sum())
    delayed = np.isin(action, ["cool_60", "hold_callback"]) & legit
    delayed_eff = (delayed * (1 - eff_wl)).sum()
    n_legit = legit.sum()
    big_legit = legit & (df.amount.values >= C.RBI_LAG_THRESHOLD)

    # ---- scam loss with incident chaining
    p_stop = _p_stop(df, action, wl)
    s = df.assign(p_stop=p_stop)[df.is_scam.values].sort_values(["incident_id", "seq_in_incident"])
    loss, prevented = 0.0, 0.0
    by_typo = {}
    for inc, g in s.groupby("incident_id"):
        alive = 1.0
        for amt, p, t in zip(g.amount.values, g.p_stop.values, g.typology.values):
            l = alive * (1 - p) * amt
            loss += l
            prevented += amt - l
            d = by_typo.setdefault(t, [0.0, 0.0])
            d[0] += amt - l
            d[1] += amt
            alive *= 1 - p * EFF.incident_break_prob
    total = loss + prevented
    return {
        "scam_value_total": total,
        "scam_value_prevented": prevented,
        "scam_value_lost": loss,
        "prevention_rate": prevented / total if total else 0,
        "prevention_by_typology": {k: v[0] / v[1] for k, v in by_typo.items()},
        "legit_friction_cost": friction,
        "net_benefit": prevented - friction,
        "legit_delayed_pct": float(delayed_eff / n_legit),
        "legit_delayed_count": float(delayed_eff),
        "legit_over10k_delayed_pct": float((delayed * (1 - eff_wl))[big_legit].sum() / big_legit.sum()),
        "legit_nudged_pct": float(((action == "nudge") & legit).sum() / n_legit),
        "legit_blocked_pct": float(((action == "block") & legit).sum() / n_legit),
        "action_mix_legit": pd.Series(action[legit]).value_counts(normalize=True).to_dict(),
        "action_mix_scam": pd.Series(action[~legit]).value_counts(normalize=True).to_dict(),
        "n_legit": int(n_legit), "n_scam": int((~legit).sum()),
    }


def tune_thresholds(df: pd.DataFrame, p: np.ndarray, max_delay_pct: float = 0.01) -> Thresholds:
    """Grid-search expected-loss thresholds to maximise net benefit, subject to
    delaying at most `max_delay_pct` of legitimate payments."""
    best, best_nb = Thresholds(), -np.inf
    grid = itertools.product([10, 25, 100, 300], [150, 300, 500, 1_500],
                             [700, 1_000, 2_000, 5_000, 15_000], [0.85, 0.95, 0.99])
    for n, c, h, b in grid:
        if not n < c < h:
            continue
        th = Thresholds(el_nudge=n, el_cool=c, el_hold=h, p_block=b)
        r = evaluate_policy(df, decide_batch(df, p, th).action.values)
        if r["legit_delayed_pct"] <= max_delay_pct and r["net_benefit"] > best_nb:
            best, best_nb = th, r["net_benefit"]
    return best


def compare(test: pd.DataFrame, p: np.ndarray, th: Thresholds) -> dict:
    out = {}
    out["P0_status_quo"] = evaluate_policy(test, np.full(len(test), "allow", dtype=object))
    a, wl = flat_lag_actions(test, C.RBI_LAG_THRESHOLD)
    out["P1_rbi_flat_10k"] = evaluate_policy(test, a, wl)
    a, wl = flat_lag_actions(test, C.BANKS_LAG_THRESHOLD)
    out["P2_flat_25k"] = evaluate_policy(test, a, wl)
    dec = decide_batch(test, p, th)
    out["P3_smart_cooling_off"] = evaluate_policy(test, dec.action.values)
    out["P3_smart_cooling_off"]["guardrail_hits"] = dec.rule.value_counts().to_dict()
    return out


def stress_test(test: pd.DataFrame, p: np.ndarray, th: Thresholds,
                noise_levels=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0), seed: int = 7) -> list[dict]:
    """Degrade the model by adding Gaussian noise to the logit of p, then re-run
    the Smart policy with the *same* thresholds. Shows how much of the benefit
    survives a much weaker model than the one trained on synthetic data."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(seed)
    y = test.is_scam.values
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    out = []
    for s in noise_levels:
        pn = 1 / (1 + np.exp(-(logit + rng.normal(0, s, len(p)))))
        r = evaluate_policy(test, decide_batch(test, pn, th).action.values)
        out.append({"noise_sigma": s, "roc_auc": float(roc_auc_score(y, pn)),
                    "pr_auc": float(average_precision_score(y, pn)),
                    "prevention_rate": r["prevention_rate"], "legit_delayed_pct": r["legit_delayed_pct"],
                    "legit_blocked_pct": r["legit_blocked_pct"], "net_benefit": r["net_benefit"]})
    return out
