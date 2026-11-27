# Solution design: Smart Cooling-Off

## 1. Problem statement
APP scams are payments the customer authorises while being deceived. Because UPI and IMPS settle in
seconds, the window to stop or recover the money is extremely short. NCRP data cited by RBI shows
reported cases rising from 2.6 lakh in 2021 to 28 lakh in 2025, and value from ₹551 crore to
₹22,931 crore. RBI's April 2026 discussion paper proposes four options:

| # | RBI option | Design response in this prototype |
|---|---|---|
| 1 | 1-hour lag on APP ≥ ₹10k, whitelisting allowed | Replaced by risk-tiered `cool_60` / `hold_callback`; no whitelist bypass for high-risk payments |
| 2 | Trusted-person authentication: 70+/PwD, ≥ ₹50k | Guard-rail G3, risk-gated so low-risk payments by seniors are not held |
| 3 | Annual credit cap of ₹25 lakh with "shadow credits" | Out of scope (payee bank); payee-side mule signals are used as features instead |
| 4 | Customer controls / kill switch | Complementary; a kill-switch state can be fed in as a hard rule |

## 2. Target users and KPIs
- **Primary:** retail customers making P2P transfers; vulnerable segments get extra protection.
- **Bank owners:** Fraud Risk, Digital Channels, Customer Experience, Compliance.
- **KPIs:** scam value prevented (%), genuine payments delayed (%), false-block rate, call-back
  volume (operations capacity), complaint rate, p99 decision latency under 50 ms.

## 3. Architecture (payer bank)
1. **Payment initiation hook.** The mobile/net-banking app or the bank's PSP switch calls
   `/v1/decide` before releasing the UPI request to NPCI.
2. **Feature assembly**, kept under 20 ms:
   - Customer profile (CBS/CRM): age, tenure, vulnerability flag, typical amounts.
   - Session/device (mobile SDK): new device, screen-share or remote-access app, SIM-swap flag
     (telco API).
   - Call context: on-call, duration, unknown/VoIP caller. This needs a consented telco integration
     or OS-level signals.
   - Payee intelligence: MuleHunter.AI score, CFCFRMS/1930 reports, DPIP/IDPIC feeds, account age,
     fan-in, pass-through behaviour.
   - Velocity (streaming feature store, e.g. Flink/Kafka + Redis): counts and amounts over 1h,
     24h and 7d.
3. **Scoring.** A gradient-boosted model with isotonic calibration, served in-process or via ONNX.
   SHAP contributions become reason codes.
4. **Policy engine.** An expected-loss ladder plus guard-rails. Thresholds are versioned config
   owned by Fraud Risk, not hard-coded.
5. **Orchestration of interventions:**
   - `cool_60`: provisional debit, cancel API, push/SMS. Re-screen the payee at T+15 and T+45 min
     against fresh DPIP/MuleHunter signals.
   - `hold_callback`: create a fraud-ops case; outbound call (IVR first, then an agent) or a
     trusted-person approval request; auto-release or auto-cancel rules on SLA expiry.
   - `block`: decline with a 1930 referral; if CFCFRMS-linked, report back to I4C.
6. **Feedback loop.** Outcomes flow back as labels for retraining: fraud reports, cancellations,
   call-back verdicts and chargebacks.

## 4. Responsible AI (RBI FREE-AI alignment)
- **Explainability:** every non-allow decision carries the top-3 reason codes, written for
  customers and for the call centre.
- **Fairness:** age and disability are used *protectively*, in line with RBI Option 2. Monitor
  delay and block rates by age band, gender and city tier, and alert on disparity.
- **Human in the loop:** irreversible actions (block) are limited to high-confidence or
  intelligence-confirmed cases; holds always end with a human or customer decision.
- **Governance:** model inventory entry, board-approved AI policy, drift monitoring (PSI on top
  features), quarterly validation, and AI incident reporting.
- **Privacy:** only consented telco/device signals are used; collect the minimum needed; follow
  DPDP Act 2023 obligations.

## 5. Rollout plan
1. **Shadow mode (6–8 weeks):** score live traffic and log the decisions without acting. Calibrate
   effectiveness and friction assumptions.
2. **Pilot:** nudges only, then `cool_60` for the top 0.5% expected loss, with a 10% holdout.
3. **Scale:** add `hold_callback` once call-centre capacity is sized from pilot volumes. Add the
   G2/G3 guard-rails.
4. **Ecosystem:** share payee-risk outcomes back to DPIP/IDPIC and MuleHunter (federated where
   possible).

## 6. Open research questions
- Measuring causal effectiveness of each intervention by typology (uplift modelling).
- Privacy-preserving cross-bank payee risk (federated learning, secure aggregation).
- Detecting coaching from interaction telemetry (typing cadence, app switching) when telco signals
  are unavailable.
- Optimal delay length as a function of payee-side detection latency.
