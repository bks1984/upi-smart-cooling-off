"""Gradient-boosted scam-risk model with calibration and per-transaction reason codes."""
from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config as C
from .features import FEATURES, REASON_TEXT, build

PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=40,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    scale_pos_weight=20, verbose=-1, seed=C.SEED,
)


@dataclass
class RiskModel:
    booster: lgb.Booster
    calibrator: IsotonicRegression

    def raw(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(X[FEATURES])

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated probability that the payment is a scam."""
        return self.calibrator.predict(self.raw(X))

    def reasons(self, X: pd.DataFrame, top_k: int = 3) -> list[list[str]]:
        """Top-k positive SHAP contributions -> human-readable reason codes."""
        contrib = self.booster.predict(X[FEATURES], pred_contrib=True)[:, :-1]
        out = []
        for row in contrib:
            idx = np.argsort(-row)[:top_k]
            out.append([REASON_TEXT[FEATURES[i]] for i in idx if row[i] > 0])
        return out

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "RiskModel":
        return joblib.load(path)


def split(tx: pd.DataFrame):
    cut = tx.ts.min().normalize() + pd.Timedelta(days=C.TRAIN_DAYS)
    return tx[tx.ts < cut].copy(), tx[tx.ts >= cut].copy()


def train(train_df: pd.DataFrame) -> tuple[RiskModel, dict]:
    # inner out-of-time validation: last 10 days of the training window
    vcut = train_df.ts.max().normalize() - pd.Timedelta(days=10)
    tr, va = train_df[train_df.ts < vcut], train_df[train_df.ts >= vcut]
    Xtr, Xva = build(tr), build(va)
    dtr = lgb.Dataset(Xtr, tr.is_scam.astype(int))
    dva = lgb.Dataset(Xva, va.is_scam.astype(int), reference=dtr)
    booster = lgb.train(PARAMS, dtr, num_boost_round=1500, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    cal.fit(booster.predict(Xva), va.is_scam.astype(int))
    info = {"best_iteration": booster.best_iteration, "n_train": len(tr), "n_valid": len(va)}
    return RiskModel(booster, cal), info


def evaluate(model: RiskModel, df: pd.DataFrame) -> dict:
    X = build(df)
    y = df.is_scam.astype(int).values
    p = model.score(X)
    raw = model.raw(X)
    order = np.argsort(-raw)
    res = {
        "roc_auc": float(roc_auc_score(y, raw)),
        "pr_auc": float(average_precision_score(y, raw)),
        "base_rate": float(y.mean()),
        "n_txn": int(len(y)), "n_scam": int(y.sum()),
    }
    amt = df.amount.values
    for rate in (0.005, 0.01, 0.02, 0.05):
        k = int(rate * len(y))
        sel = order[:k]
        res[f"recall_at_{rate:.1%}_alerts"] = float(y[sel].sum() / y.sum())
        res[f"value_recall_at_{rate:.1%}_alerts"] = float((amt[sel] * y[sel]).sum() / (amt * y).sum())
    imp = model.booster.feature_importance("gain")
    res["feature_importance_gain"] = dict(sorted(
        {f: float(g) for f, g in zip(FEATURES, imp / imp.sum())}.items(), key=lambda kv: -kv[1]))
    return res
