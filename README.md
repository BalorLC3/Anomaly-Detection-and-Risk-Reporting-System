# Anomaly Detection & Automated Risk Reporting System

A production-structured pipeline for detecting abnormal behavioral and risk patterns
in historical data, with automated natural-language report generation via
local LLMs (DeepSeek via Ollama) and RAG-based contextual enrichment.

---

## Project Structure

```
anomaly-risk-system/
├── config.py                   # Central configuration (paths, model, thresholds)
│
├── detection/
│   ├── detector.py             # MultiClassGaussianAnomalyDetector
│   └── features.py             # Feature engineering utilities
│
├── evaluation/
│   └── metrics.py              # AUC, AP, FPR, cross-val, per-class breakdown
│
├── reporting/
│   ├── knowledge_base.py       # FAISS RAG knowledge base
│   ├── prompts.py              # LangChain prompt templates
│   └── generator.py            # AnomalyReportGenerator (LLM + RAG)
│
├── models/                     # Saved detector .joblib files
├── knowledge_base/             # Persisted FAISS index
├── reports/                    # Output JSON reports
│
└── notebook/
    └── main_pipeline.ipynb     # Orchestration notebook
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Ollama and pull DeepSeek
```bash
ollama serve
ollama pull deepseek-r1:8b
```

### 3. Run the notebook
Open `notebook/main_pipeline.ipynb` and run cells top to bottom.
The notebook covers:
- Feature engineering
- Detector training + cross-validation
- ROC-AUC / Precision-Recall / FPR evaluation
- RAG knowledge base construction
- Automated report generation

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Diagonal Mahalanobis | Keeps per-feature Z-score explanations interpretable |
| Robust estimators (trimmed mean, IQR std) | Reduces sensitivity to training-set noise |
| Empirical thresholds | Matches actual score distribution vs. theoretical chi-square |
| Per-class models | Captures category-specific normal behaviour |
| FAISS + Ollama embeddings | Fully local — no external API calls |
| DeepSeek via Ollama | Local LLM, no data leaves the machine |

---

## Evaluation Results

| Metric | Score |
|---|---|
| ROC-AUC | 0.976 |
| Average Precision | 0.543 |

---

## Requirements

- Python 3.12+
- [Ollama](https://ollama.com) running locally
- Model pulled: `ollama pull deepseek-r1:8b`
