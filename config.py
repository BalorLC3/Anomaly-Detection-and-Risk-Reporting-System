"""
config.py — Central configuration for the anomaly detection system.
All paths, model parameters, and LLM settings live here.
"""

from pathlib import Path

# -- Paths --
ROOT_DIR          = Path(__file__).parent
MODELS_DIR        = ROOT_DIR / "models"
KNOWLEDGE_BASE_DIR = ROOT_DIR / "knowledge_base"
REPORTS_DIR       = ROOT_DIR / "reports"

# Create directories if they don't exist
for _dir in (MODELS_DIR, KNOWLEDGE_BASE_DIR, REPORTS_DIR):
    _dir.mkdir(exist_ok=True)

# -- Detector --
DETECTOR_CONTAMINATION = 0.016   # expected anomaly rate 
DETECTOR_MIN_SAMPLES   = 30      # minimum class samples to fit a per-class model

# -- Evaluation --
CV_N_SPLITS  = 3      # cross-validation folds
CV_RANDOM_STATE = 17

# -- LLM (Ollama / DeepSeek) --
OLLAMA_MODEL       = "deepseek-r1:8b"   
OLLAMA_TEMPERATURE = 0.2                # low = consistent, factual output

# -- RAG --
RAG_TOP_K = 3   # number of similar historical cases to retrieve per query

# -- Reporting --
MAX_INDIVIDUAL_REPORTS = 20   # cap on per-anomaly LLM calls per pipeline run
