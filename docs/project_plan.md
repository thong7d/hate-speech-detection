# Project Plan and Management

## 1. Project Schedule and Milestones
The development of the ViHSD content moderation platform is organized into distinct sequential phases:

```
[ Phase 1: Problem Definition & Data Sourcing ] -> [ Phase 2: Exploratory Data Analysis & Baseline ]
                                                                       |
[ Phase 4: Model Deployment (FastAPI & Docker) ] <- [ Phase 3: Fine-Tuning XLM-R Classifier ]
                     |
[ Phase 5: Streamlit & Gradio GUI Offline Serving ] -> [ Phase 6: Qwen2.5 Agentic Integration & Audit ]
```

| Phase | Milestone | Deliverable | Duration |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Data Sourcing | Raw CSV downloads, processing scripts | Week 1-2 |
| **Phase 2** | Baseline Modeling | TF-IDF + Logistic Regression baseline | Week 3 |
| **Phase 3** | Deep Learning Core | Fine-tuned XLM-RoBERTa base model | Week 4-5 |
| **Phase 4** | Server Infrastructure | FastAPI endpoints & Docker containerization | Week 6 |
| **Phase 5** | Interactive GUI | Local Streamlit and Gradio dashboards | Week 7 |
| **Phase 6** | Agent Integration | Ollama Qwen2.5 borderline routing & logging | Week 8 |

---

## 2. Task Breakdown

### A. Data Engineering
- Sourcing ViHSD dataset.
- Developing cleaning regex filters and word segmentations in `src/data/preprocessing.py`.
- Converting datasets into optimized Parquet formats.

### B. Machine Learning & Modeling
- Setting up the HuggingFace training scripts (`notebooks/train_colab.ipynb`).
- Fine-tuning the XLM-R base model.
- Saving predictions, reports, and confusion matrices to `results/`.

### C. MLOps & Software Engineering
- Developing REST API using FastAPI (`src/api/`).
- Containerizing the API with Docker and docker-compose.
- Creating the Local GUI in Streamlit (`streamlit_app.py`) and Gradio (`app.py`).
- Establishing the thread-safe registry (`InferenceProgressRegistry`) for background batch inference.
- Refactoring `src/agent/moderator.py` to utilize local Ollama LLM endpoints.

---

## 3. Team Organization and Roles (Conceptual)
For a typical implementation team:
- **ML/NLP Engineer**: Responsible for model training, hyperparameter optimization, and fine-tuning.
- **Backend Developer**: Builds the FastAPI service, integrations, and handles Docker deployment.
- **Frontend / UI Engineer**: Implements Streamlit & Gradio user dashboards.
- **Data Scientist**: Analyzes evaluation reports, error trends, and validates adversarial test sets.

---

## 4. Reflection on Team Scaling
If the team size scales up from a solo developer to a 5-10 person engineering unit, roles would segment as follows:
- **Dedicated Data Annotators**: Curate and tag new datasets captured by the agent feedback loops.
- **Infrastructure / MLOps Engineer**: Builds the automated retraining orchestrations, continuous deployment pipelines, and manages local GPU clusters for Ollama services.
- **Responsible AI / Ethics Lead**: Audits outputs, conducts systematic bias checks, and regulates PII filtering methods.
