# Ethics & Responsible AI Statement

## 1. Ethics Impact Statement
Automated text classification systems used for moderation represent a gatekeeping force. They directly impact freedom of speech, digital expression, and online community dynamics. While filtering hate speech is vital for maintaining safe communities, over-filtering or biased classifications can silence legitimate discussions, political critique, and minority dialects.

---

## 2. Stakeholder Benefit & Harm Analysis
- **Beneficiaries**:
  - **Online Users**: Protected from hate speech, psychological harm, and targeted harassment.
  - **Marginalized Groups**: Toxic campaigns targeting vulnerable groups are filtered out.
  - **Social Media Businesses**: Reduction in moderation overhead and brand safety risks.
- **Harm Risks**:
  - **False Positives**: Users sharing legitimate opinions, sarcasm, or reclaimed words could be silenced, causing frustration and feelings of censorship.
  - **Regional Dialects**: Dialect bias may result in higher false-positive rates for Southern or Central Vietnamese speakers whose slang might be incorrectly categorized.

---

## 3. Bias and Fairness Mitigation
To ensure fairness and prevent systematic bias:
1. **Cascade Routing Thresholds**: The baseline model uses calibrated class thresholds (`{"CLEAN": 0.5, "OFFENSIVE": 0.38, "HATE": 0.32}`) rather than raw argmax. The lower threshold for HATE ensures high recall for toxic comments, while the agentic routing step acts as a second-pass validator to reduce false positives.
2. **Representative Validation**: Validation is performed on balanced dev sets checking performance across regional dialects and slang variants.
3. **Appeals Workflow**: An explicit user appeals process should accompany the deployment, allowing users to dispute automated classification results, feeding the corrections back to the human-in-the-loop pipeline.

---

## 4. Model Explainability for Non-Technical Stakeholders
Deep learning models are often criticised as "black boxes". We improve transparency through:
- **Soft Probability Outputs**: Displaying probability distributions in the GUI (via plotly charts) rather than binary labels. This helps stakeholders understand when the model is uncertain.
- **Agentic Explanations**: When the Ollama Qwen2.5 agent is triggered, it generates a natural language explanation (in Vietnamese) clarifying the context, linguistic cues, and semantic nuances behind the final classification.

---

## 5. Potential Misuse and Censorship Concerns
- **State Censorship**: The tool could be configured by oppressive actors to flag political criticism or dissent under the guise of filtering toxic content.
- **Corporate Bias**: Platforms might use it to suppress unionization efforts or consumer criticism.
- **Mitigation**: The system's taxonomy strictly defines HATE as targeted hostility towards protected characteristics (race, gender, sexual orientation, religion) and OFFENSIVE as general vulgarity, separating them from harmless political or business debates.
