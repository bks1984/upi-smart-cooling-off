import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scooler import config as C  # noqa: E402
from scooler.data_generator import generate  # noqa: E402
from scooler.policy import Thresholds, decide_batch  # noqa: E402
from scooler.simulate import evaluate_policy, flat_lag_actions  # noqa: E402


def _row(**kw):
    base = dict(amount=20_000.0, age=40, is_disabled=False, payee_is_new=True, on_active_call=False,
                caller_unknown=False, call_duration_min=0.0, payee_reported_cfcfrms=False,
                payee_mule_score=0.05)
    base.update(kw)
    return pd.DataFrame([base])


TH = Thresholds()


def test_low_risk_allowed():
    assert decide_batch(_row(), np.array([0.0001]), TH).action[0] == "allow"


def test_trusted_payee_never_delayed_when_low_risk():
    d = decide_batch(_row(payee_is_new=False, amount=900_000), np.array([0.01]), TH)
    assert d.action[0] == "allow" and d.rule[0] == "trusted_payee"


def test_expected_loss_tiers():
    assert decide_batch(_row(amount=10_000), np.array([0.05]), TH).action[0] == "nudge"     # EL 500
    assert decide_batch(_row(amount=100_000), np.array([0.05]), TH).action[0] == "cool_60"  # EL 5k
    assert decide_batch(_row(amount=1_000_000), np.array([0.05]), TH).action[0] == "hold_callback"


def test_coached_victim_escalates_to_callback():
    r = _row(on_active_call=True, caller_unknown=True, call_duration_min=120, amount=50_000)
    d = decide_batch(r, np.array([0.03]), TH)
    assert d.action[0] == "hold_callback" and d.rule[0] == "coached_victim_callback"


def test_rbi_option2_vulnerable():
    d = decide_batch(_row(age=74, amount=60_000), np.array([0.02]), TH)
    assert d.action[0] == "hold_callback"


def test_cfcfrms_hard_block():
    d = decide_batch(_row(payee_reported_cfcfrms=True, payee_mule_score=0.9), np.array([0.001]), TH)
    assert d.action[0] == "block"


@pytest.fixture(scope="module")
def small_data():
    C.N_CUSTOMERS, C.SIM_DAYS = 2_000, 20
    _, tx = generate(seed=1)
    return tx


def test_generator_has_all_typologies(small_data):
    assert set(small_data.typology.unique()) >= {"legit", "investment"}
    assert small_data.is_scam.mean() < 0.02


def test_status_quo_prevents_nothing(small_data):
    r = evaluate_policy(small_data, np.full(len(small_data), "allow", dtype=object))
    assert r["scam_value_prevented"] == 0 and r["legit_friction_cost"] == 0


def test_flat_lag_scope(small_data):
    a, _ = flat_lag_actions(small_data, C.RBI_LAG_THRESHOLD)
    assert (a[small_data.amount.values < C.RBI_LAG_THRESHOLD] == "allow").all()
