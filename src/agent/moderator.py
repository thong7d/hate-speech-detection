import os
import json
import time
import hashlib
from datetime import datetime
from typing import Optional

from langdetect import detect as langdetect_detect
import google.generativeai as genai

try:
    from evaluation.classifier import load_hf_artifacts, predict_with_artifacts
except ImportError:
    from src.evaluation.classifier import load_hf_artifacts, predict_with_artifacts


# ========== TOOL IMPLEMENTATIONS ==========

class ModerationTools:
    """
    Collection of tools available to the Gemini agent.
    Each tool has a clear input/output contract.
    """

    def __init__(self, model_source: str, log_path: str, device: str = "auto"):
        """
        Args:
            model_source: HF model ID or local path to the fine-tuned model.
            log_path: Path to the JSONL log file on Drive.
            device: 'cuda', 'cpu', or 'auto'.
        """
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        hf_token = os.environ.get("HF_TOKEN")
        self.artifacts = load_hf_artifacts(
            model_source,
            token=hf_token,
            device=device,
            max_length=int(os.environ.get("MAX_LENGTH", "128")),
        )
        self.device = self.artifacts.device
        self.borderline_low = float(os.environ.get("BORDERLINE_LOW", "0.35"))
        self.borderline_high = float(os.environ.get("BORDERLINE_HIGH", "0.65"))
        print(f"✅ ModerationTools initialized on {self.device}")

    def classify_text(self, text: str) -> dict:
        """
        Tool 1: Classify text for hate speech using the fine-tuned model.

        Input:  text (str) — the text to classify
        Output: dict with keys: label, label_id, confidence, scores, is_borderline
        """
        result = predict_with_artifacts(
            self.artifacts,
            text,
            borderline_low=self.borderline_low,
            borderline_high=self.borderline_high,
            preprocess=True,
        )
        return {
            "label": result["label"],
            "label_id": result["label_id"],
            "confidence": result["confidence"],
            "scores": result["scores"],
            "probabilities": result["probabilities"],
            "is_borderline": result["is_borderline"],
        }

    def detect_language(self, text: str) -> dict:
        """
        Tool 2: Detect the language of the input text.

        Input:  text (str)
        Output: dict with key: language (ISO 639-1 code)
        """
        try:
            lang = langdetect_detect(text)
        except Exception:
            lang = "unknown"
        return {"language": lang}

    def log_event(self, text: str, label: str, action: str, reason: str,
                  confidence: float = 0.0, language: str = "vi"):
        """
        Tool 3: Log a moderation event to JSONL file (privacy-safe).

        Input:  text (for hashing only), label, action, reason, confidence, language
        Output: Event appended to JSONL log file. Returns confirmation dict.

        Privacy: Only the SHA-256 hash prefix of the text is stored, not the raw text.
        """
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "text_hash": text_hash,
            "text_length": len(text),
            "language": language,
            "label": label,
            "confidence": confidence,
            "action": action,
            "reason": reason,
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

        return {"status": "logged", "event_hash": text_hash}


# ========== GEMINI AGENT ==========

SYSTEM_PROMPT = """You are a content moderation agent for a Vietnamese social platform.
You have access to 3 tools:

1. classify_text(text: str) → Returns toxicity classification
   {label: CLEAN/OFFENSIVE/HATE, confidence: float, scores: dict}

2. detect_language(text: str) → Returns language code (vi, en, ...)

3. log_event(text_hash: str, label: str, action: str, reason: str)
   → Logs moderation decision (privacy-safe: hash only)

Your reasoning process:
Step 1: Detect the language of the input.
Step 2: Classify the text using classify_text().
Step 3: Reason about the result:
  - If CLEAN with high confidence (>0.7): allow, explain why.
  - If confidence is low (0.3–0.65) or result is OFFENSIVE/HATE
    but the text seems ambiguous: ask ONE specific clarifying question
    tailored to the content (do NOT use generic questions).
  - If HATE with high confidence (>0.7): block and explain clearly.
Step 4: After final decision, call log_event().
Step 5: Respond to the user in Vietnamese.

Always explain your reasoning briefly to the user.
When asking clarifying questions, be specific about what aspect of the text is ambiguous.
"""


class ContentModerator:
    """
    Gemini-powered content moderation agent with tool usage.

    Implements multi-step reasoning:
    1. Receive user text
    2. Use tools to classify and analyze
    3. Make moderation decision (allow / ask clarification / block)
    4. Log the decision
    """

    def __init__(self, tools: ModerationTools, gemini_api_key: str,
                 gemini_model: str = "gemini-1.5-flash"):
        """
        Args:
            tools: ModerationTools instance with loaded model.
            gemini_api_key: Google Gemini API key.
            gemini_model: Gemini model name (default: gemini-1.5-flash).
        """
        self.tools = tools

        genai.configure(api_key=gemini_api_key)
        self.gemini = genai.GenerativeModel(gemini_model)
        self.conversation_history = []

        print(f"✅ ContentModerator initialized with {gemini_model}")

    def moderate(self, user_text: str, context: Optional[str] = None) -> str:
        """
        Run the full moderation pipeline on a piece of text.

        Input:
            user_text: The text to moderate.
            context: Optional additional context (e.g., user's clarification).

        Output:
            str — The agent's response in Vietnamese.
        """
        # Step 1: Detect language
        lang_result = self.tools.detect_language(user_text)
        # Step 2: Classify text
        cls_result = self.tools.classify_text(user_text)

        # Build prompt for Gemini
        tool_results = (
            f"Language detection result: {json.dumps(lang_result)}\n"
            f"Classification result: {json.dumps(cls_result)}"
        )

        if context:
            user_message = (
                f"Original text to moderate: \"{user_text}\"\n"
                f"User provided additional context: \"{context}\"\n"
                f"Tool results:\n{tool_results}\n\n"
                f"Based on the classification results and the user's context, "
                f"make your final moderation decision."
            )
        else:
            user_message = (
                f"Text to moderate: \"{user_text}\"\n"
                f"Tool results:\n{tool_results}\n\n"
                f"Analyze the classification results and make your moderation decision."
            )

        # Call Gemini
        try:
            chat = self.gemini.start_chat(history=[])
            response = chat.send_message(
                f"{SYSTEM_PROMPT}\n\n{user_message}"
            )
            agent_response = response.text
        except Exception as e:
            agent_response = (
                f"⚠️ Agent error: {str(e)}. "
                f"Falling back to direct classification: "
                f"{cls_result['label']} (confidence: {cls_result['confidence']})"
            )

        # Step 4: Log the event
        action = "ALLOW" if cls_result["label"] == "CLEAN" else (
            "REVIEW" if cls_result["is_borderline"] else "BLOCK"
        )
        self.tools.log_event(
            text=user_text,
            label=cls_result["label"],
            action=action,
            reason=f"confidence={cls_result['confidence']}",
            confidence=cls_result["confidence"],
            language=lang_result["language"],
        )

        return agent_response

    def reset_conversation(self):
        """Reset conversation history for a new moderation session."""
        self.conversation_history = []
