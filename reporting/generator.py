"""
AnomalyReportGenerator

Connects detector output → RAG retrieval → DeepSeek/Qwen (via Ollama) →
structured report dict.

Two entry points:
  explain_anomaly()      — single-row natural-language explanation
  run_full_pipeline()    — executive summary + per-anomaly explanations
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

import config
from reporting.knowledge_base import AnomalyKnowledgeBase
from reporting.prompts import SINGLE_ANOMALY_PROMPT, SUMMARY_REPORT_PROMPT


class AnomalyReportGenerator:
    """
    Generates natural-language reports for anomalies detected by
    MultiClassGaussianAnomalyDetector.

    Parameters
    ----------
    model_name     : Ollama model tag (default: config.OLLAMA_MODEL).
    knowledge_base : Fitted AnomalyKnowledgeBase for RAG context retrieval.
    temperature    : LLM sampling temperature (default: 0.2 — factual/consistent).
    """

    def __init__(
        self,
        model_name: str = None,
        knowledge_base: Optional[AnomalyKnowledgeBase] = None,
        temperature: float = None,
    ):
        model_name  = model_name  or config.OLLAMA_MODEL
        temperature = temperature or config.OLLAMA_TEMPERATURE

        self.llm = OllamaLLM(model=model_name, temperature=temperature)
        self.kb  = knowledge_base

        self._single_chain  = SINGLE_ANOMALY_PROMPT | self.llm | StrOutputParser()
        self._summary_chain = SUMMARY_REPORT_PROMPT | self.llm | StrOutputParser()

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _format_risk_factors(explanation: Dict) -> str:
        factors = explanation.get("risk_factors", [])
        if not factors:
            return "No significant individual feature deviations detected."
        lines = [
            f"  • {f['feature']}: {f['direction']} than normal "
            f"(value={f['value']:.3f}, deviation={f['deviation_std']:.2f}σ)"
            for f in factors
        ]
        return "\n".join(lines)

    @staticmethod
    def _build_rag_query(category: str, explanation: Dict) -> str:
        features = [f["feature"] for f in explanation.get("risk_factors", [])]
        return f"anomaly in {category} with high {', '.join(features[:3])}"

    # -- Single anomaly -----------------------------------------------------

    def explain_anomaly(self, row: pd.Series, category_col: str = "category") -> str:
        """
        Generate a natural-language explanation for a single anomalous row.

        Parameters
        ----------
        row          : A row from the DataFrame returned by detector.predict().
        category_col : Name of the category column in the result DataFrame.
        """
        explanation: Dict = row.get("explanation", {})
        category: str     = str(row.get(category_col, "Unknown"))

        rag_query = self._build_rag_query(category, explanation)
        historical_context = self.kb.retrieve(rag_query) if self.kb else "N/A"

        result = self._single_chain.invoke({
            "category":           category,
            "anomaly_score":      float(row.get("anomaly_score", 0)),
            "threshold":          float(row.get("threshold", 0)),
            "severity":           explanation.get("severity", "unknown"),
            "risk_factors":       self._format_risk_factors(explanation),
            "historical_context": historical_context,
        })
        return result.strip()

    # -- Executive summary --------------------------------------------------

    def generate_summary_report(
        self,
        result_df: pd.DataFrame,
        category_col: str = "category",
        period: Optional[str] = None,
    ) -> str:
        """
        Generate an executive summary for an entire detection run.

        Parameters
        ----------
        result_df    : Full output of detector.predict().
        category_col : Name of the category column.
        period       : Label for the reporting period (default: today's date).
        """
        period    = period or datetime.now().strftime("%Y-%m-%d")
        anomalies = result_df[result_df["is_anomaly"]]

        total_records   = len(result_df)
        total_anomalies = len(anomalies)
        anomaly_rate    = total_anomalies / max(total_records, 1)

        # Per-category breakdown
        cat_counts = anomalies[category_col].value_counts()
        cat_total  = result_df[category_col].value_counts()
        cat_lines  = [
            f"  • {cat}: {cnt} anomalies ({cnt / cat_total.get(cat, 1):.1%} of category)"
            for cat, cnt in cat_counts.items()
        ]
        category_breakdown = "\n".join(cat_lines) or "No anomalies detected."

        # Most frequent risk features
        feature_counts: Dict[str, int] = {}
        for _, row in anomalies.iterrows():
            expl = row.get("explanation", {})
            for f in expl.get("risk_factors", []) if isinstance(expl, dict) else []:
                feature_counts[f["feature"]] = feature_counts.get(f["feature"], 0) + 1

        sorted_features = sorted(feature_counts.items(), key=lambda x: -x[1])
        top_features_str = "\n".join(
            f"  • {feat}: appeared in {cnt} anomalies"
            for feat, cnt in sorted_features[:10]
        ) or "No recurring features identified."

        # RAG retrieval with top systemic features
        top_feat_names = [f for f, _ in sorted_features[:3]]
        rag_query = (
            f"systemic risk pattern: {', '.join(top_feat_names)}"
            if top_feat_names else "anomaly risk summary"
        )
        historical_context = self.kb.retrieve(rag_query, k=4) if self.kb else "N/A"

        result = self._summary_chain.invoke({
            "period":             period,
            "total_records":      total_records,
            "total_anomalies":    total_anomalies,
            "anomaly_rate":       anomaly_rate,
            "category_breakdown": category_breakdown,
            "top_risk_features":  top_features_str,
            "historical_context": historical_context,
        })
        return result.strip()

    # -- Full pipeline ------------------------------------------------------

    def run_full_pipeline(
        self,
        result_df: pd.DataFrame,
        category_col: str = "category",
        max_individual_reports: int = None,
        period: Optional[str] = None,
    ) -> Dict:
        """
        Run the complete reporting pipeline:
          1. Executive summary for the full detection run.
          2. Individual LLM explanations for each anomaly, sorted by severity
             (high → medium → low) then by anomaly score descending.

        Parameters
        ----------
        result_df               : Full output of detector.predict().
        category_col            : Name of the category column.
        max_individual_reports  : Cap on per-anomaly LLM calls (default: config value).
        period                  : Reporting period label.

        Returns
        -------
        dict with keys: metadata, summary, anomaly_reports
        """
        max_individual_reports = max_individual_reports or config.MAX_INDIVIDUAL_REPORTS
        period                 = period or datetime.now().strftime("%Y-%m-%d")

        print("[Pipeline] Generating executive summary...")
        summary = self.generate_summary_report(result_df, category_col, period)

        # Sort anomalies: severity first, then highest score
        severity_order = {"high": 0, "medium": 1, "low": 2}
        anomalies = result_df[result_df["is_anomaly"]].copy()
        anomalies["_sev_order"] = anomalies["explanation"].apply(
            lambda e: severity_order.get(e.get("severity", "low"), 2)
            if isinstance(e, dict) else 2
        )
        anomalies = (
            anomalies
            .sort_values(["_sev_order", "anomaly_score"], ascending=[True, False])
            .head(max_individual_reports)
        )

        individual_reports = []
        total = len(anomalies)

        for i, (idx, row) in enumerate(anomalies.iterrows(), 1):
            print(f"[Pipeline] Explaining anomaly {i}/{total} (index={idx})...")
            explanation_text = self.explain_anomaly(row, category_col)
            expl_dict = row.get("explanation", {})
            individual_reports.append({
                "index":         idx,
                "category":      row.get(category_col, "N/A"),
                "anomaly_score": float(row.get("anomaly_score", 0)),
                "threshold":     float(row.get("threshold", 0)),
                "severity":      expl_dict.get("severity", "unknown") if isinstance(expl_dict, dict) else "unknown",
                "risk_factors":  expl_dict.get("risk_factors", [])   if isinstance(expl_dict, dict) else [],
                "llm_explanation": explanation_text,
            })

        return {
            "metadata": {
                "period":             period,
                "total_records":      len(result_df),
                "total_anomalies":    int(result_df["is_anomaly"].sum()),
                "reports_generated":  len(individual_reports),
                "model":              self.llm.model,
            },
            "summary":         summary,
            "anomaly_reports": individual_reports,
        }

    # -- Persistence --------------------------------------------------------

    def save_report(self, report: Dict, path: str | Path) -> None:
        """Persist the full report dict to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[Pipeline] Report saved → {path}")
