"""End-to-end pipeline: generate data -> train -> tune policy -> compare -> report.

Run from repo root:  python scripts/run_pipeline.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from scooler import config as C  # noqa: E402
from scooler import model as M  # noqa: E402
from scooler.data_generator import generate  # noqa: E402
from scooler.features import build  # noqa: E402
from scooler.policy import decide_batch, warning_for  # noqa: E402
from scooler.reporting import make_charts  # noqa: E402
from scooler.simulate import compare, stress_test, tune_thresholds  # noqa: E402


def main():
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)

    print("1/5 Generating synthetic data ...")
    cust, tx = generate()
    cust.to_csv(ROOT / "data/customers.csv.gz", index=False)
    tx.to_csv(ROOT / "data/transactions.csv.gz", index=False)
    print(f"    {len(tx):,} APP transfers, {tx.is_scam.sum():,} scam payments, "
          f"{tx[tx.is_scam].incident_id.nunique()} incidents")

    print("2/5 Training risk model (out-of-time split) ...")
    train_df, test_df = M.split(tx)
    model, info = M.train(train_df)
    model.save(str(ROOT / "models/risk_model.joblib"))
    metrics = M.evaluate(model, test_df)
    metrics.update(info)

    print("3/5 Tuning intervention thresholds on validation window ...")
    vcut = train_df.ts.max().normalize() - np.timedelta64(10, "D")
    val = train_df[train_df.ts >= vcut]
    th = tune_thresholds(val, model.score(build(val)), max_delay_pct=0.01)
    print("    ", th.to_dict())

    print("4/5 Comparing policies on test window ...")
    p_test = model.score(build(test_df))
    results = compare(test_df, p_test, th)
    for k, r in results.items():
        print(f"    {k:24s} prevented={r['prevention_rate']:.1%}  legit delayed={r['legit_delayed_pct']:.2%}"
              f"  net benefit=Rs {r['net_benefit']/1e5:,.1f} lakh")

    stress = stress_test(test_df, p_test, th)
    for r in stress:
        print(f"    stress sigma={r['noise_sigma']}: AUC={r['roc_auc']:.3f} PR-AUC={r['pr_auc']:.3f} "
              f"prevented={r['prevention_rate']:.1%} delayed={r['legit_delayed_pct']:.2%}")

    # sample decisions with reason codes for the README / dashboard
    dec = decide_batch(test_df, p_test, th)
    sample_idx = []
    for a in ["block", "hold_callback", "cool_60", "nudge"]:
        cand = test_df.index[(dec.action == a).values]
        sample_idx += list(cand[:2])
    Xs = build(test_df.loc[sample_idx])
    reasons = model.reasons(Xs)
    samples = []
    for i, r in zip(sample_idx, reasons):
        row = test_df.loc[i]
        samples.append({
            "txn_id": int(row.txn_id), "amount": float(row.amount), "age": int(row.age),
            "payee_is_new": bool(row.payee_is_new), "p_scam": float(p_test[test_df.index.get_loc(i)]),
            "action": dec.action[i], "rule": dec.rule[i], "reasons": r,
            "warning": warning_for(row), "truth": row.typology,
        })

    print("5/5 Writing reports ...")
    out = {"model_metrics": metrics, "thresholds": th.to_dict(), "policies": results, "stress_test": stress,
           "sample_decisions": samples,
           "assumptions": {"friction_cost": C.FRICTION_COST, "victim_share": C.VICTIM_SHARE,
                           "effectiveness": C.Effectiveness().table,
                           "coached_whitelist_prob": C.COACHED_WHITELIST_PROB}}
    (ROOT / "reports/results.json").write_text(json.dumps(out, indent=2, default=float))
    make_charts(out, ROOT / "reports")
    print("Done. See reports/results.json and reports/*.png")


if __name__ == "__main__":
    main()
