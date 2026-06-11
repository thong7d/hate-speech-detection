# Continual Learning & Monitoring Strategy

## 1. Overview
Automated content moderation systems operate in dynamic environments where language, slang, and toxic targets evolve rapidly. To prevent model degradation, a systematic plan for monitoring and continual learning is established.

---

## 2. Monitoring Metrics and Drift Detection
We implement real-time monitoring of inference activities to detect drift:

### A. Core Metrics
- **Routing Ratio (RR)**: The fraction of comments routed to the LLM agent:
  $$\text{Routing Ratio} = \frac{\text{Comments Routed to Agent}}{\text{Total Comments Processed}}$$
  - **Drift Indicator**: A sudden spike in RR (e.g., exceeding **35%**) suggests that the distribution of incoming text is drifting away from the XLM-R classifier's training set, causing lower confidence levels.
- **Label Distribution Drift**: Daily monitoring of output label shares (CLEAN vs. OFFENSIVE vs. HATE). Significant shifts suggest potential domain drift.
- **Model Confidence Distribution**: Monitoring shifts in the mean and median confidence scores of the classifier.

### B. Mitigation Actions
- If `Routing Ratio` is consistently higher than 30% for a 7-day window, a system alert is triggered requesting model inspection.
- The administrator is prompted via the GUI to adjust the confidence routing thresholds (e.g., lower BORDERLINE threshold limits).

---

## 3. Data Collection and Feedback Loop
To gather high-value samples for future retraining:

1. **Agent Audit Logs**: All borderline comments routed to the agent are logged (with hashes) in `scratch/system_audit_log.jsonl`.
2. **Human-in-the-Loop (HITL) Curation**: Administrators review samples logged as `borderline` through the GUI or custom dashboard. Correct labels are annotated.
3. **User Feedback Flagging**: Platform end-users can flag misclassified items (false negatives or false positives), creating a secondary correction pipeline.

---

## 4. Retraining Strategy
Once sufficient new data is gathered, retraining occurs in three steps:

```
[ New Curated & Annotated Data ]
             |
             v
[ Active Learning / Data Augmentation ]
             |
             v
[ Automated Retraining Pipeline (Makefile: train) ]
             |
             v
[ Offline Golden Test Suite Evaluation (F1 & Accuracy) ]
             |
             v
[ Gatekeeper Stage: Compare new vs old model ]
             |
             v
[ Canary Deployment / Green-Blue Release ]
```

### Gatekeeper Rules:
Retrained models must satisfy the following conditions before promotion to production:
1. Macro F1 must not degrade on the golden test set (threshold: $\ge 64.61\%$).
2. Latency P95 must remain within SLAs ($\le 30$ms).
3. The routing ratio on a simulated validation batch must remain below 15%.
