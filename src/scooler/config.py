"""
Central configuration and *explicit assumptions* for the Smart Cooling-Off engine.

Every number here that is not a regulatory parameter is a modelling ASSUMPTION
for a synthetic-data prototype. They are kept in one place so a bank team can
replace them with values estimated from its own data.

Regulatory anchors (RBI Discussion Paper "Exploring safeguards in digital
payments to curb frauds", DPSS, 9 April 2026):
  * Option 1 - one-hour lag on authorised push payments (APP) >= Rs 10,000,
    at the payer bank, with transaction/payee-level whitelisting; merchant,
    recurring and cheque payments exempt.
  * Option 2 - "trusted person" additional authentication for APP >= Rs 50,000
    by customers aged 70+ and persons with disabilities.
  * Banks' feedback (May 2026): support a lag but at a Rs 25,000 threshold.
"""
from dataclasses import dataclass, field

SEED = 42

# ---------------------------------------------------------------- simulation
N_CUSTOMERS = 20_000
SIM_DAYS = 60
TRAIN_DAYS = 40            # out-of-time split: first 40 days train, last 20 test
P2P_RATE_PER_DAY = 0.30    # mean account-to-account transfers per customer per day

# Victim incidence is INFLATED vs. reality so the model has enough positives
# to learn from. Real-world prevalence is far lower; see README "Limitations".
VICTIM_SHARE = 0.012

# Typology mix of incidents (counts, not value). Value mix emerges from the
# amount distributions and is calibrated loosely to I4C 2025 loss shares
# (investment ~77%, digital arrest ~8%).
TYPOLOGY_MIX = {
    "investment": 0.34,
    "digital_arrest": 0.08,
    "job_task": 0.20,
    "kyc_impersonation": 0.20,
    "sextortion": 0.18,
}

# ---------------------------------------------------------------- regulation
RBI_LAG_THRESHOLD = 10_000
BANKS_LAG_THRESHOLD = 25_000
TRUSTED_PERSON_THRESHOLD = 50_000
SENIOR_AGE = 70


@dataclass
class Effectiveness:
    """P(scam payment is stopped | intervention, typology).

    Encodes the key insight from public critique of a flat lag: coached victims
    (e.g. 'digital arrest', where fraudsters stay on a video call for hours) are
    barely helped by a passive timer, but are helped by an *active* bank
    call-back / trusted-person check.
    """
    table: dict = field(default_factory=lambda: {
        #                 nudge  cool_60  hold_callback  block
        # block < 1.0: coached victims often retry via another bank / app
        "investment":        (0.10, 0.35, 0.55, 0.75),
        "digital_arrest":    (0.03, 0.10, 0.50, 0.60),
        "job_task":          (0.12, 0.35, 0.55, 0.75),
        "kyc_impersonation": (0.20, 0.45, 0.65, 0.85),
        "sextortion":        (0.05, 0.15, 0.30, 0.60),
    })
    # During any delay, payee-side intelligence (MuleHunter / DPIP) may flag the
    # beneficiary. Extra stop probability per hour of delay, scaled by mule score.
    payee_intel_boost_per_hour: float = 0.15
    # If one payment in an incident is stopped, the victim is alerted and later
    # payments in that incident are prevented with this probability.
    incident_break_prob: float = 0.70

    def p_stop(self, typology: str, action: str) -> float:
        idx = {"nudge": 0, "cool_60": 1, "hold_callback": 2, "block": 3}
        if action not in idx:
            return 0.0
        return self.table[typology][idx[action]]


# Probability a *coached* victim whitelists the fraudster's payee when the bank
# offers a whitelist bypass (flat-lag policies only).
COACHED_WHITELIST_PROB = {
    "investment": 0.50, "digital_arrest": 0.65, "job_task": 0.40,
    "kyc_impersonation": 0.30, "sextortion": 0.35,
}
# Share of legitimate repeat payees a typical customer whitelists.
LEGIT_WHITELIST_SHARE = 0.60

# Friction cost (Rs-equivalent) imposed on a *legitimate* payment per action:
# customer inconvenience + abandonment + bank operations cost.
FRICTION_COST = {
    "allow": 0.0,
    "nudge": 5.0,
    "cool_60": 150.0,
    "hold_callback": 560.0,   # Rs 500 inconvenience + Rs 60 call-centre cost
    "block": 1500.0,          # false block: complaint, possible churn
}
DELAY_MINUTES = {"allow": 0, "nudge": 0, "cool_60": 60, "hold_callback": 240, "block": 0}

ACTIONS = ["allow", "nudge", "cool_60", "hold_callback", "block"]
