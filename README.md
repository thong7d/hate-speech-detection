# ViHSD: Vietnamese Hate Speech & Toxicity Detection System

An end-to-end, production-ready Vietnamese hate speech and toxicity detection system built on a fine-tuned **XLM-RoBERTa Base** backbone and integrated with a local **Ollama Qwen2.5 Agent** for advanced reasoning on borderline cases.

This repository serves as a university final project submission for NLP and MLOps.

---

## 📖 Project Documentation Index
All project documentation is written in English and is available in the `docs/` folder:

1. **[Business Problem Definition](file:///d:/000MINHTHONG/Junior%20-%20Semester%20II/ANLPB/FinalProject/hate-speech-detection/docs/problem_definition.md)**: Defines business context, stakeholders, system scope, and success metrics.
2. **[Data Description Document](file:///d:/000MINHTHONG/Junior%20-%20Semester%20II/ANLPB/FinalProject/hate-speech-detection/docs/data_description.md)**: Details dataset splits, data schema, preprocessing steps, and class imbalances.
3. **[Agentic AI Architecture](file:///d:/000MINHTHONG/Junior%20-%20Semester%20II/ANLPB/FinalProject/hate-speech-detection/docs/agent_architecture.md)**: Outlines the multi-step reasoning flow, tool configurations, system prompts, and contains a Mermaid architecture diagram.
4. **[Model Evaluation and Analysis](file:///d:/000MINHTHONG/Junior%20-%20Semester%20II/ANLPB/FinalProject/hate-speech-detection/docs/model_evaluation.md)**: Compares the TF-IDF+LR baseline against the XLM-R classifier, includes the classification report, confusion matrix, and system trade-offs.
5. **[System Deployment Document](file:///d:/000MINHTHONG/Junior%20-%20Semester%20II/ANLPB/FinalProject/hate-speech-detection/docs/deployment.md)**: Details FastAPI endpoints, local GUI serving (Streamlit + Gradio), and Docker container configuration.
6. **[Data Privacy & Model Robustness](file:///d:/000MINHTHONG/Junior%20-%20Semester%20II/ANLPB/FinalProject/hate-speech-detection/docs/privacy_analysis.md)**: Analyzes PII handling, cryptographic logging (SHA-256), and adversarial attack robustness.
7. **[Ethics & Responsible AI Statement](file:///d:/000MINHTHONG/Junior%20-%20Semester%20II/ANLPB/FinalProject/hate-speech-detection/docs/ethics_statement.md)**: Evaluates algorithmic fairness, explainability, regional biases, and mitigation of potential misuses.
8. **[Project Plan and Management](file:///d:/000MINHTHONG/Junior%20-%20Semester%20II/ANLPB/FinalProject/hate-speech-detection/docs/project_plan.md)**: Documents project schedule, timeline milestones, and reflections on team scaling.

---

## 🛠️ System Architecture

```text
Incoming Vietnamese Text
  │
  ▼
XLM-RoBERTa Base Classifier ─────► [Confidence >= 0.65] ──► Output (Fast Classify)
  │
  ▼ [Confidence < 0.65] (Grey Area)
ContentModerator Agent
  ├─► Tool 1: detect_language (langdetect)
  ├─► Tool 2: classify_text (XLM-R probabilities)
  ├─► LLM Reasoning: Ollama Qwen2.5-7B (Ngrok Tunnel/Colab)
  └─► Tool 3: log_event (Privacy-safe SHA-256 logging)
  │
  ▼
Final Decision Label & Vietnamese Explanation
```

---

## 🚀 Getting Started

### 1. Local Installation
Clone the repository and install dependencies:
```bash
# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
.venv\Scripts\activate     # On Windows

# Install packages
pip install -r requirements.txt
```

### 2. Environment Variables Configuration
Copy the env template and edit it with your keys:
```bash
cp .env.example .env
```
Ensure you set the correct URL to your Ollama service (e.g., local server or Colab Ngrok tunnel).

---

## 🖥️ Running the Services

### 1. FastAPI Web Server
Run the REST API locally:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```
API docs will be available at `http://localhost:8000/docs`.

### 2. Local Streamlit GUI
Run the admin operations dashboard:
```bash
streamlit run streamlit_app.py
```
- Features a single text tester and thread-safe batch CSV processor.
- Displays dynamic charts and live Routing Ratio metrics.

### 3. Gradio Demo Interface
Run the interactive Gradio demo:
```bash
python app.py
```
Serves the web page on `http://localhost:7860`.

### 4. Running with Docker Compose
To deploy the API in a containerized environment:
```bash
docker compose up --build
```

---

## 🧪 Testing and Validation
Run the unit test suite to check config, preprocessing, model loading, and API routing:
```bash
python -m pytest tests/
```
All tests should pass, showing correct class configurations and model fallbacks.
