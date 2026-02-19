"""
All LangChain prompt templates used by AnomalyReportGenerator.

Keeping prompts in one place makes them easy to tune independently
of the generation logic.
"""

from langchain_core.prompts import PromptTemplate

# ── Single anomaly explanation ─────────────────────────────────────────────

SINGLE_ANOMALY_PROMPT = PromptTemplate(
    input_variables=[
        "category",
        "anomaly_score",
        "threshold",
        "severity",
        "risk_factors",
        "historical_context",
    ],
    template="""You are a risk analyst reviewing a transaction flagged by an automated fraud detection system.
Write a concise, professional explanation suitable for inclusion in a risk report.

## Flagged Transaction
- Category     : {category}
- Anomaly Score: {anomaly_score:.3f}  (threshold: {threshold:.3f})
- Severity     : {severity}

## Top Contributing Risk Factors
{risk_factors}

## Relevant Historical Cases
{historical_context}

## Instructions
1. In 2–3 sentences, explain why this transaction is anomalous.
2. Identify the most critical risk factors and their significance.
3. Reference any relevant patterns from the historical cases above.
4. Recommend one specific next action: investigate, escalate, monitor, or dismiss.
5. Keep the total response under 200 words. Be direct — no filler phrases.

Report:""",
)


# ── Executive summary for a full detection run ────────────────────────────

SUMMARY_REPORT_PROMPT = PromptTemplate(
    input_variables=[
        "period",
        "total_records",
        "total_anomalies",
        "anomaly_rate",
        "category_breakdown",
        "top_risk_features",
        "historical_context",
    ],
    template="""You are a senior risk analyst writing an executive summary for a periodic fraud risk report.

## Detection Run — {period}
- Total records scored : {total_records}
- Anomalies detected   : {total_anomalies}  ({anomaly_rate:.1%} of total)

## Breakdown by Category
{category_breakdown}

## Most Frequent Risk Features Across All Anomalies
{top_risk_features}

## Relevant Historical Context
{historical_context}

## Instructions
Write a professional executive summary (300–400 words) covering:
1. Overall risk posture for this period.
2. Which categories require immediate attention and why.
3. Whether patterns in the risk features appear systemic or isolated.
4. Comparison with historical cases — normal variation or a new trend?
5. Recommended actions, prioritised by urgency.

Be direct and specific. Avoid generic statements.

Executive Summary:""",
)
