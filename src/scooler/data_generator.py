"""
Synthetic generator for Indian account-to-account (APP) transfers with
realistic scam typologies.

Typologies are modelled on public reporting of Indian cyber-financial fraud
(I4C / NCRP 2025 data, RBI DPSS discussion paper April 2026):

* investment        - fake trading / IPO / crypto apps; escalating payments to
                      rotating mule accounts over days. Largest share of loss.
* digital_arrest    - impersonated police/CBI/RBI on long video calls; few very
                      large transfers, often elderly victims, high balance drain.
* job_task          - "like & earn" / task scams; many small, escalating payments.
* kyc_impersonation - fake bank / customer-care calls; screen-share or remote-
                      access apps; 1-2 payments.
* sextortion        - threats after video calls; repeated moderate payments.

All data is synthetic. No real customer data is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def _customers(rng: np.random.Generator, n: int) -> pd.DataFrame:
    age = np.clip(rng.gamma(9.0, 4.4, n) + 12, 18, 90).astype(int)
    income_band = rng.choice([1, 2, 3, 4], size=n, p=[0.35, 0.35, 0.22, 0.08])
    median_amt = np.array([0, 1_000, 2_500, 6_000, 15_000])[income_band]
    balance = rng.lognormal(np.log(np.array([0, 25e3, 90e3, 3e5, 1.2e6])[income_band]), 0.8)
    lit = np.clip(rng.beta(4, 2, n) - (age > 65) * 0.25 + (age < 30) * 0.1, 0, 1)
    return pd.DataFrame({
        "customer_id": np.arange(n),
        "age": age,
        "is_disabled": rng.random(n) < 0.02,
        "gender_m": rng.random(n) < 0.55,
        "tenure_months": rng.integers(1, 240, n),
        "city_tier": rng.choice([1, 2, 3], n, p=[0.4, 0.35, 0.25]),
        "income_band": income_band,
        "median_amt": median_amt,
        "balance": balance,
        "digital_literacy": lit.round(3),
    })


def _legit(rng, cust: pd.DataFrame) -> pd.DataFrame:
    n_per = rng.poisson(C.P2P_RATE_PER_DAY * C.SIM_DAYS, len(cust))
    idx = np.repeat(cust.index.values, n_per)
    m = len(idx)
    c = cust.loc[idx].reset_index(drop=True)

    day = rng.uniform(0, C.SIM_DAYS, m)
    hour = np.clip(rng.normal(14, 4.0, m), 0, 23.99)
    ts = pd.Timestamp("2026-08-01") + pd.to_timedelta(np.floor(day), "D") + pd.to_timedelta(hour, "h")

    is_new = rng.random(m) < 0.28
    # mixture: everyday transfers + occasional large legit (rent deposit, tuition,
    # family emergency, property token) -> hard negatives
    big = rng.random(m) < 0.04
    amount = np.where(
        big,
        rng.lognormal(np.log(c.median_amt * 12 + 20_000), 0.7),
        rng.lognormal(np.log(c.median_amt), 0.9),
    )
    amount = np.clip(np.round(amount, -1), 10, 2e7)

    on_call = rng.random(m) < 0.10
    biz = is_new & (rng.random(m) < 0.12)   # small traders, landlords, tutors
    df = pd.DataFrame({
        "ts": ts,
        "customer_id": c.customer_id,
        "payee_id": np.where(is_new, "N" + pd.Series(np.arange(m)).astype(str),
                             "K" + c.customer_id.astype(str) + "_" + rng.integers(0, 8, m).astype(str)),
        "amount": amount,
        "payee_is_new": is_new,
        "on_active_call": on_call,
        "call_duration_min": np.where(on_call, rng.exponential(6, m), 0.0),
        "caller_unknown": on_call & (rng.random(m) < 0.25),
        "caller_voip_or_intl": on_call & (rng.random(m) < 0.04),
        "video_call": on_call & (rng.random(m) < 0.10),
        "screen_share_active": rng.random(m) < 0.005,
        "remote_access_app": rng.random(m) < 0.003,
        "new_device_7d": rng.random(m) < 0.03,
        "sim_swap_7d": rng.random(m) < 0.002,
        "remark_risk_kw": rng.random(m) < 0.05,
        "payee_account_age_days": np.where(is_new, rng.lognormal(np.log(700), 1.3, m),
                                           rng.lognormal(np.log(2000), 0.7, m)),
        "payee_fan_in_24h": rng.poisson(np.where(biz, 3.0, np.where(is_new, 0.6, 0.2))),
        "payee_passthrough_ratio": np.where(biz, rng.beta(4, 2, m), rng.beta(1.5, 5, m)),
        "payee_mule_score": np.where(biz, rng.beta(2, 5, m), rng.beta(1.2, 10, m)),
        "payee_reported_cfcfrms": rng.random(m) < 0.0008,
        "is_scam": False,
        "typology": "legit",
        "incident_id": -1,
        "seq_in_incident": 0,
    })
    df["balance_before"] = c.balance.values * rng.uniform(0.6, 1.4, m)
    return df


_TYPO_AGE_WEIGHT = {
    "investment": lambda a: np.exp(-((a - 42) / 12) ** 2),
    "digital_arrest": lambda a: 0.2 + (a >= 55) * 1.0 + (a >= 70) * 1.5,
    "job_task": lambda a: np.exp(-((a - 26) / 7) ** 2),
    "kyc_impersonation": lambda a: 0.4 + (a >= 50) * 1.0,
    "sextortion": lambda a: np.exp(-((a - 28) / 9) ** 2),
}


def _mule_pool(rng, n=1_500) -> pd.DataFrame:
    evaded = rng.random(n) < 0.35       # mules that MuleHunter-style models miss
    return pd.DataFrame({
        "payee_id": "M" + pd.Series(np.arange(n)).astype(str),
        # ~half are "rented" older accounts of real people, which look normal
        "payee_account_age_days": np.where(rng.random(n) < 0.5, rng.lognormal(np.log(60), 1.0, n),
                                           rng.lognormal(np.log(700), 1.0, n)),
        "payee_mule_score": np.where(evaded, rng.beta(1.5, 8, n), rng.beta(4, 3, n)),
        "payee_reported_cfcfrms": rng.random(n) < 0.12,
        "payee_passthrough_ratio": rng.beta(3, 2, n),
        "fan_in_mu": rng.uniform(0.2, 4, n),
    })


def _scams(rng, cust: pd.DataFrame) -> pd.DataFrame:
    mules = _mule_pool(rng)
    n_victims = int(C.VICTIM_SHARE * len(cust))
    typos = rng.choice(list(C.TYPOLOGY_MIX), n_victims, p=list(C.TYPOLOGY_MIX.values()))
    rows, used = [], set()
    for inc_id, t in enumerate(typos):
        w = _TYPO_AGE_WEIGHT[t](cust.age.values).astype(float)
        if t == "sextortion":
            w = w * np.where(cust.gender_m, 1.0, 0.3)
        w[list(used)] = 0
        v = int(rng.choice(len(cust), p=w / w.sum()))
        used.add(v)
        cu = cust.iloc[v]
        start = pd.Timestamp("2026-08-01") + pd.Timedelta(days=float(rng.uniform(0, C.SIM_DAYS - 3)))

        if t == "investment":
            n = int(rng.integers(3, 10)); a0 = rng.lognormal(np.log(12_000), 0.6); g = 1.55
            gaps_h = rng.uniform(10, 96, n)
        elif t == "digital_arrest":
            n = int(rng.integers(1, 3)); a0 = rng.lognormal(np.log(1.6e5), 0.8); g = 1.2
            gaps_h = rng.uniform(0.3, 20, n)
        elif t == "job_task":
            n = int(rng.integers(4, 11)); a0 = rng.lognormal(np.log(1_500), 0.5); g = 1.55
            gaps_h = rng.uniform(0.5, 30, n)
        elif t == "kyc_impersonation":
            n = int(rng.integers(1, 3)); a0 = rng.lognormal(np.log(35_000), 0.8); g = 1.1
            gaps_h = rng.uniform(0.05, 1, n)
        else:  # sextortion
            n = int(rng.integers(2, 6)); a0 = rng.lognormal(np.log(12_000), 0.6); g = 1.3
            gaps_h = rng.uniform(2, 48, n)

        t_cur = start
        bal = cu.balance * rng.uniform(0.9, 1.6)
        inc_mules = mules.sample(int(rng.integers(1, 4)), random_state=int(rng.integers(1e9)))
        for k in range(n):
            if k:
                t_cur = t_cur + pd.Timedelta(hours=float(gaps_h[k]))
            amt = float(np.round(min(a0 * g ** k, 1.5e6) * rng.uniform(0.8, 1.2), -1))
            if t == "digital_arrest":   # victims break FDs / borrow: drain most of balance
                amt = max(amt, bal * rng.uniform(0.35, 0.8))
            m = inc_mules.iloc[k % len(inc_mules)]
            long_call = t in ("digital_arrest", "kyc_impersonation")
            on_call = rng.random() < {"investment": 0.15, "digital_arrest": 0.97, "job_task": 0.10,
                                      "kyc_impersonation": 0.88, "sextortion": 0.45}[t]
            hour_shift = rng.normal(0, 5) if t == "digital_arrest" else rng.normal(0, 3)
            ts = t_cur.normalize() + pd.Timedelta(hours=float(np.clip(t_cur.hour + hour_shift, 0, 23.9)))
            rows.append({
                "ts": ts, "customer_id": cu.customer_id, "payee_id": m.payee_id,
                "amount": max(amt, 100.0),
                "payee_is_new": (k < len(inc_mules)),
                "on_active_call": on_call,
                "call_duration_min": (rng.uniform(45, 600) if long_call else rng.exponential(15)) if on_call else 0.0,
                "caller_unknown": on_call and rng.random() < 0.92,
                "caller_voip_or_intl": on_call and rng.random() < (0.7 if long_call else 0.35),
                "video_call": on_call and rng.random() < {"digital_arrest": 0.85, "sextortion": 0.6}.get(t, 0.1),
                "screen_share_active": rng.random() < (0.6 if t == "kyc_impersonation" else 0.04),
                "remote_access_app": rng.random() < (0.45 if t == "kyc_impersonation" else 0.02),
                "new_device_7d": rng.random() < (0.15 if t == "kyc_impersonation" else 0.04),
                "sim_swap_7d": rng.random() < (0.05 if t == "kyc_impersonation" else 0.005),
                "remark_risk_kw": rng.random() < {"investment": 0.25, "digital_arrest": 0.20,
                                                  "job_task": 0.25}.get(t, 0.10),
                "payee_account_age_days": m.payee_account_age_days,
                "payee_fan_in_24h": rng.poisson(m.fan_in_mu),
                "payee_passthrough_ratio": m.payee_passthrough_ratio,
                "payee_mule_score": m.payee_mule_score,
                "payee_reported_cfcfrms": m.payee_reported_cfcfrms,
                "is_scam": True, "typology": t, "incident_id": inc_id, "seq_in_incident": k,
                "balance_before": bal,
            })
            bal = max(bal - amt * 0.7, bal * 0.2)  # victims top up from savings / loans
    return pd.DataFrame(rows)


def _apply_signal_coverage(rng, df: pd.DataFrame) -> pd.DataFrame:
    """Not every bank/customer has every signal. Telco call-context (via a
    consented telco API) is observed for ~55% of sessions; device-level app
    detection (screen-share / remote-access) for ~70%. Unobserved -> 0."""
    m = len(df)
    tel = rng.random(m) < 0.55
    for col in ["on_active_call", "caller_unknown", "caller_voip_or_intl", "video_call"]:
        df[col] = df[col] & tel
    df["call_duration_min"] = np.where(tel, df["call_duration_min"], 0.0)
    dev = rng.random(m) < 0.70
    for col in ["screen_share_active", "remote_access_app"]:
        df[col] = df[col] & dev
    df["telco_signal_available"] = tel
    return df


def _velocity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["customer_id", "ts"]).reset_index(drop=True)
    g = df.set_index("ts").groupby("customer_id")
    df["payer_txn_count_1h"] = g["amount"].rolling("1h").count().values - 1
    df["payer_txn_count_24h"] = g["amount"].rolling("24h").count().values - 1
    df["payer_amt_24h"] = g["amount"].rolling("24h").sum().values - df["amount"].values
    tmp = df.assign(new_amt=df.amount * df.payee_is_new).set_index("ts").groupby("customer_id")
    df["payer_new_payee_amt_7d"] = tmp["new_amt"].rolling("7D").sum().values - df["amount"] * df["payee_is_new"]
    return df


def generate(seed: int = C.SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    cust = _customers(rng, C.N_CUSTOMERS)
    tx = pd.concat([_legit(rng, cust), _scams(rng, cust)], ignore_index=True)
    tx = tx.merge(cust[["customer_id", "age", "is_disabled", "tenure_months", "city_tier",
                        "digital_literacy", "median_amt"]], on="customer_id")
    tx = _apply_signal_coverage(rng, tx)
    tx = _velocity(tx)
    tx["hour"] = tx.ts.dt.hour + tx.ts.dt.minute / 60
    tx["balance_drain_ratio"] = np.clip(tx.amount / tx.balance_before, 0, 1.5)
    tx["amount_to_median_ratio"] = tx.amount / tx.median_amt
    # A repeat legit payee is whitelistable; mules only if the victim is coached
    tx["payee_repeat"] = ~tx.payee_is_new
    tx = tx.sort_values("ts").reset_index(drop=True)
    tx.insert(0, "txn_id", np.arange(len(tx)))
    return cust, tx
