# Data Privacy & Model Robustness Analysis

## 1. Personally Identifiable Information (PII) Scrubbing
Social media comments often contain sensitive personal data (e.g. phone numbers, bank accounts, home addresses, real names).
- **Strategy**: Implementing pre-processing regex filters in the moderation pipeline to strip or mask PII (e.g., converting phone numbers to `[PHONE_NUM]`).
- **Anonymization in Storage**: Permanent system logs MUST NOT contain raw comments. The logging utility hashes texts using SHA-256 and only stores the first 16 characters of the hash alongside metadata.

```python
# PII scrubbing regex configuration
import re
PHONE_REGEX = re.compile(r'\b(0[3|5|7|8|9])+([0-9]{8})\b')

def scrub_pii(text: str) -> str:
    # Replace phone numbers
    text = PHONE_REGEX.sub("[PHONE]", text)
    return text
```

---

## 2. Cryptographic Audit Logging
To allow tracing while respecting user privacy:
- The system logs decisions in a `.jsonl` audit file.
- The log fields include: `timestamp`, `text_hash` (first 16 hex chars of `SHA-256(text)`), `text_length`, `language`, `label`, `confidence`, and `action`.
- Since SHA-256 is a one-way function, the original text cannot be reconstructed from the log, preventing information leaks if logs are compromised.

---

## 3. Credential Security
- **No Hardcoded Credentials**: API Keys (e.g., HuggingFace token, Ollama endpoint domains) are stored in local environment variables.
- **Git Protections**: `.env` is declared in `.gitignore` to prevent committing secrets to GitHub. A template file `.env.example` is committed instead.

---

## 4. Model Robustness against Adversarial Attacks
Hate speech detection models are targets for adversarial evasion (e.g., deliberate typos, character insertion, word spacing tricks).

### A. Typographical Adversaries
Common tricks include replacing characters (e.g., `đờ mờ` -> `đ.ờ m.ờ`) or using homoglyphs.
- **XLM-R Classifier Robustness**: Fine-tuned on social media text which naturally contains noisy spellings. It utilizes a subword tokenizer (byte-level BPE) which preserves semantic information for slightly corrupted words.
- **Agentic Fallback**: The Qwen2.5 agent utilizes deep semantic understanding of contextual slang and character substitutions to recognize hidden slurs.

### B. Sarcasm and Aggression without Bad Words
- **Example**: *"Lũ này học thức cao siêu quá nên mới phá nát xã hội."* (Sarcastic hate targeting a group, containing only clean words).
- The baseline XLM-R might classify this as `CLEAN` due to the lack of vulgar vocabulary.
- However, because the baseline model predicts this with borderline confidence, the agentic router intercepts the message and corrects the label to `HATE` based on deep semantic context.

---

## 5. Adversarial Testing Suite
The robustness is evaluated programmatically in `src/training/robustness_cases.py` and validated by unit tests in `tests/test_robustness_cases.py`, ensuring stability against known attacks.
