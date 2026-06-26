"""
Ollama Qwen2.5-powered content moderation agent with tool usage.

Implements the Agentic AI component:
  1. Multi-step reasoning via Qwen2.5 LLM
  2. Tool usage: classify_text, detect_language, log_event
  3. Dynamic decision-making: borderline texts get analyzed by LLM
"""
import os
import json
import hashlib
import requests
from datetime import datetime
from typing import Optional, List, Dict, Any

from langdetect import detect as langdetect_detect

try:
    from src.models.classifier import HateSpeechClassifier
except ImportError:
    from models.classifier import HateSpeechClassifier


# ========== TOOL IMPLEMENTATIONS ==========

class ModerationTools:
    """
    Collection of tools available to the moderation agent.
    Each tool has a clear input/output contract.
    """

    def __init__(self, classifier: Optional[HateSpeechClassifier] = None, 
                 model_source: str = "huggingface", 
                 hf_repo_id: str = "thong7d/vihsd-xlmr-base-hate-speech",
                 log_path: str = "scratch/system_audit_log.jsonl", 
                 device: str = "auto",
                 use_word_segmentation: bool = False):
        """
        Args:
            classifier: Pre-loaded HateSpeechClassifier instance. If None, it will be loaded.
            model_source: 'huggingface' or 'local'.
            hf_repo_id: HF model ID or local path to the fine-tuned model.
            log_path: Path to the JSONL log file.
            device: 'cuda', 'cpu', or 'auto'.
            use_word_segmentation: Whether to use Vietnamese word segmentation.
        """
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        if classifier is not None:
            self.classifier = classifier
        else:
            self.classifier = HateSpeechClassifier(
                model_source=model_source,
                hf_repo_id=hf_repo_id,
                device=device,
                use_word_segmentation=use_word_segmentation,
            )

        self.borderline_low = float(os.environ.get("BORDERLINE_LOW", "0.35"))
        self.borderline_high = float(os.environ.get("BORDERLINE_HIGH", "0.65"))
        print(f"✅ ModerationTools initialized with classifier on {self.classifier.device}")

    def classify_text(self, text: str) -> dict:
        """
        Tool 1: Classify text for hate speech using the fine-tuned XLM-R model.

        Input:  text (str) — the text to classify
        Output: dict with keys: label, confidence, scores, is_borderline
        """
        result = self.classifier.predict(text)
        confidence = result["confidence"]
        
        # Get class probabilities
        probs = result.get("probabilities", {})
        probs_raw = result.get("probabilities_raw", probs)
        
        # Decoupled minority class routing
        p_offensive_raw = probs_raw.get("OFFENSIVE", 0.0)
        p_hate_raw = probs_raw.get("HATE", 0.0)
        
        p_offensive_contrib = p_offensive_raw if p_offensive_raw >= 0.01 else 0.0
        p_hate_contrib = p_hate_raw if p_hate_raw >= 0.01 else 0.0
        p_toxic_raw = p_offensive_contrib + p_hate_contrib
        
        is_borderline = p_toxic_raw >= 0.15

        return {
            "label": result["label"],
            "confidence": confidence,
            "scores": probs,
            "probabilities": probs,
            "probabilities_raw": probs_raw,
            "is_borderline": is_borderline,
            "toxicity_score": result.get("toxicity_score") or 0.0,
            "toxic_spans": result.get("toxic_spans") or [],
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
                  confidence: float = 0.0, language: str = "vi", agent_processed: str = "NO"):
        """
        Tool 3: Log a moderation event to JSONL file (privacy-safe).

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
            "agent_processed": agent_processed
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

        return {"status": "logged", "event_hash": text_hash}


# ========== AGENT IMPLEMENTATION ==========

SYSTEM_PROMPT_SINGLE = (
    "Bạn là chuyên gia kiểm duyệt nội dung của hệ thống ViHSD. Hãy phân tích sắc thái ngữ nghĩa "
    "(mỉa mai, châm biếm, từ lóng, ngữ cảnh) của văn bản đầu vào và kết quả phân loại từ mô hình cơ sở. "
    "Hãy đưa ra quyết định cuối cùng. Chỉ trả về chuỗi JSON duy nhất theo định dạng bắt buộc, không kèm lời dẫn:\n"
    '{"final_label": "CLEAN"|"OFFENSIVE"|"HATE", "explanation": "Lý do ngắn gọn bằng tiếng Việt"}'
)

SYSTEM_PROMPT_BATCH = (
    "Bạn là chuyên gia của hệ thống ViHSD. Hãy phân tích mảng dữ liệu đầu vào chứa các câu vùng xám "
    "và trả về mảng kết quả JSON tương ứng. Định dạng bắt buộc:\n"
    '[{"id": int, "final_label": "CLEAN"|"OFFENSIVE"|"HATE", "explanation": "Lý do ngắn gọn bằng tiếng Việt"}]'
)


class ContentModerator:
    """
    Qwen2.5-powered content moderation agent with tool usage.

    Implements multi-step reasoning:
    1. Receive user text
    2. Use tools to detect language and classify
    3. Determine if LLM agent routing is needed
    4. Call local Ollama Qwen2.5 backend for borderline reasoning
    5. Fallback if JSON format or communication fails
    6. Log the moderation decision
    """

    def __init__(self, tools: ModerationTools, ollama_endpoint: Optional[str] = None,
                 model_name: str = "qwen2.5:7b-instruct"):
        """
        Args:
            tools: ModerationTools instance.
            ollama_endpoint: URL endpoint for the Ollama server.
            model_name: Name of the Ollama model.
        """
        self.tools = tools
        self.ollama_endpoint = ollama_endpoint or os.getenv("OLLAMA_ENDPOINT")
        self.model_name = model_name
        print(f"✅ ContentModerator initialized with endpoint: {self.ollama_endpoint}")

    def moderate(self, user_text: str, force_agent: bool = False) -> Dict[str, Any]:
        """
        Moderate a single comment. Run the full tool-based moderation pipeline.

        Returns:
            Dict containing final decision labels, confidence, probabilities, 
            and explanations.
        """
        # Step 1: Detect language
        lang_result = self.tools.detect_language(user_text)
        language = lang_result["language"]

        # Step 2: Classify text using baseline XLM-R
        cls_result = self.tools.classify_text(user_text)
        
        label = cls_result["label"]
        confidence = cls_result["confidence"]
        probs = cls_result["probabilities"]
        agent_triggered = False
        explanation = ""

        # Step 3: Determine if routing is needed (e.g. confidence < 0.65 or forced)
        should_route = force_agent or cls_result["is_borderline"]
        agent_processed_logged = "NO"
        log_reason = f"XLM-R baseline confidence={confidence:.4f}"

        if should_route and self.ollama_endpoint:
            agent_processed_logged = "YES"
            explanation = "LLM Refusal - Fallback to XLM-R"
            log_reason = explanation
            try:
                prompt_content = {
                    "text": user_text,
                    "baseline_classification": {
                        "label": label,
                        "confidence": confidence,
                        "probabilities": probs
                    },
                    "language": language
                }
                
                payload = {
                    "model": self.model_name,
                    "system": SYSTEM_PROMPT_SINGLE,
                    "prompt": json.dumps(prompt_content, ensure_ascii=False),
                    "format": "json",
                    "stream": False
                }
                
                url = f"{self.ollama_endpoint.rstrip('/')}/api/generate"
                response = requests.post(url, json=payload, timeout=120)
                
                if response.status_code == 200:
                    res_json = response.json()
                    response_text = res_json.get("response", "").strip()
                    
                    start = response_text.find('{')
                    end = response_text.rfind('}') + 1
                    if start == -1 or end == 0:
                        raise ValueError("JSON delimiters '{' and '}' not found in LLM response")
                    
                    json_str = response_text[start:end]
                    try:
                        agent_res = json.loads(json_str)
                    except json.JSONDecodeError as jde:
                        raise ValueError(f"Failed to parse LLM JSON response: {jde}") from jde
                    
                    if not isinstance(agent_res, dict) or "final_label" not in agent_res:
                        raise ValueError("LLM response JSON is missing 'final_label' key or is not a dictionary")
                        
                    final_lbl = agent_res["final_label"]
                    if final_lbl not in ["CLEAN", "OFFENSIVE", "HATE"]:
                        raise ValueError(f"LLM final_label '{final_lbl}' is invalid")
                        
                    label = final_lbl
                    explanation = agent_res.get("explanation", "Agent đã tối ưu nhãn dựa trên ngữ cảnh sâu.")
                    agent_triggered = True
                    log_reason = explanation
                else:
                    raise RuntimeError(f"Ollama API returned non-200 status code: {response.status_code}")
            except Exception as e:
                print(f"❌ Connection or processing error with Ollama Agent: {e}")
                explanation = "LLM Refusal - Fallback to XLM-R"
                log_reason = "FALLBACK_ERROR"
        
        # Step 4: Log the event
        action = "ALLOW" if label == "CLEAN" else ("REVIEW" if cls_result["is_borderline"] else "BLOCK")
        self.tools.log_event(
            text=user_text,
            label=label,
            action=action,
            reason=log_reason,
            confidence=confidence,
            language=language,
            agent_processed=agent_processed_logged
        )

        return {
            "text": user_text,
            "label": label,
            "confidence": confidence,
            "probabilities": probs,
            "agent_triggered": agent_triggered,
            "explanation": explanation,
            "toxicity_score": cls_result.get("toxicity_score"),
            "toxic_spans": cls_result.get("toxic_spans")
        }

    def moderate_batch(self, chunk_preds: List[Dict[str, Any]], borderline_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Perform batch moderation for borderline items. Modifies chunk_preds in-place.
        
        Args:
            chunk_preds: List of prediction structures, e.g. [{"text": str, "pred": dict, "agent_triggered": bool, "explanation": str}]
            borderline_items: List of borderline dicts, e.g. [{"id": int, "text": str}]
        
        Returns:
            The modified chunk_preds.
        """
        if not borderline_items or not self.ollama_endpoint:
            return chunk_preds

        try:
            prompt_str = json.dumps(borderline_items, ensure_ascii=False)
            payload = {
                "model": self.model_name,
                "system": SYSTEM_PROMPT_BATCH,
                "prompt": prompt_str,
                "format": "json",
                "stream": False
            }
            url = f"{self.ollama_endpoint.rstrip('/')}/api/generate"
            response = requests.post(url, json=payload, timeout=300)
            
            if response.status_code == 200:
                res_json = response.json()
                response_text = res_json.get("response", "").strip()
                
                agent_results = None
                try:
                    start = response_text.find('[')
                    end = response_text.rfind(']') + 1
                    if start != -1 and end != -1:
                        json_str = response_text[start:end]
                        agent_results = json.loads(json_str)
                    
                    if agent_results and isinstance(agent_results, list):
                        for res_item in agent_results:
                            if isinstance(res_item, dict) and "id" in res_item and "final_label" in res_item:
                                orig_id = res_item["id"]
                                final_lbl = res_item["final_label"]
                                if 0 <= orig_id < len(chunk_preds) and final_lbl in ["CLEAN", "OFFENSIVE", "HATE"]:
                                    chunk_preds[orig_id]["pred"]["label"] = final_lbl
                                    chunk_preds[orig_id]["explanation"] = res_item.get("explanation", "Agent đã tối ưu nhãn.")
                                    chunk_preds[orig_id]["agent_triggered"] = True
                except Exception as e:
                    print(f"❌ Parse JSON batch failed: {e}")
        except Exception as e:
            print(f"❌ Connection error to Ollama Agent in batch: {e}")
            
        return chunk_preds
