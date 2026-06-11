# Business Problem Definition

## 1. Business Context and Motivation
Online social platforms in Vietnam face an increasing volume of user-generated content daily. Among this content is a significant proportion of hate speech, toxic harassment, and offensive comments. Failing to moderate this content leads to severe consequences:
- **User Churn**: Safe users leave platforms dominated by toxic discourse.
- **Brand Reputation Damage**: Platforms acquire a reputation for toxicity, deterring advertisers and partners.
- **Legal Liabilities**: Vietnamese cybersecurity laws strictly prohibit toxic, anti-state, and defamatory contents.

Manual moderation at scale is economically unviable, slow, and exposes human moderators to mental fatigue. An automated, highly accurate Vietnamese Hate Speech Detection (ViHSD) system is necessary to filter content in real-time.

---

## 2. Target Users and Stakeholders
- **Platform Moderators**: Benefit from automated filtering of clear-cut clean/toxic comments, allowing them to focus only on highly ambiguous or controversial cases.
- **Community Managers**: Monitor overall toxicity trends and platform health.
- **Platform Users**: Enjoy a safer, more constructive digital environment.
- **Operations & MLOps Engineers**: Maintain and monitor system latency, throughput, and accuracy.

---

## 3. System Scope
The ViHSD moderation system is designed as a hybrid offline solution consisting of:
1. **REST API**: A high-throughput FastAPI service for real-time automated processing.
2. **Local GUI**: A Streamlit and Gradio based interface allowing platform administrators to inspect individual samples and perform batch csv moderation offline.
3. **Agentic Router**: A mechanism that detects borderline classifications and routes them to an LLM (Qwen2.5-7B) for deep contextual semantic analysis, optimizing the trade-off between speed and deep reasoning.

---

## 4. Success Metrics

| Dimension | Metric | Baseline (TF-IDF + LR) | Target / Performance |
| :--- | :--- | :--- | :--- |
| **Technical** | Classification Accuracy | 80.53% | **86.74%** |
| | Macro F1-Score | 62.50% | **64.61%** |
| | AUC-ROC | N/A | **87.88%** |
| | P95 Latency (Baseline Model) | ~100ms | **< 30ms (GPU)** |
| | Routing Ratio to LLM | N/A | **< 15%** (to minimize API cost/compute) |
| **Business** | Manual Moderation Workload | 100% | **Reduced by > 80%** |
| | Response SLA | Minutes/Hours | **Immediate (< 100ms)** |
| | False Positive Rate (Block Safe) | High | **< 5%** (protecting free speech) |
