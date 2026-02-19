"""
FAISS-based RAG knowledge base built from historical anomaly cases
and domain documentation.

All embeddings are generated locally via Ollama — no external API calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

import config


class AnomalyKnowledgeBase:
    """
    Vector store of historical anomaly cases and policy documents.
    Retrieved context is injected into LLM prompts before generation (RAG).

    Usage
    -----
    # Build once and persist
    kb = AnomalyKnowledgeBase()
    kb.build_from_records(historical_cases)
    kb.save(config.KNOWLEDGE_BASE_DIR)

    # Reload on subsequent runs
    kb = AnomalyKnowledgeBase().load(config.KNOWLEDGE_BASE_DIR)
    context = kb.retrieve("high fraud_score in electronics", k=3)
    """

    def __init__(self, model_name: str = None):
        model_name = model_name or config.OLLAMA_MODEL
        self.embeddings  = OllamaEmbeddings(model=model_name)
        self.vectorstore: Optional[FAISS] = None

    # -- Builders -----------------------------------------------------------

    def build_from_records(self, records: List[Dict]) -> "AnomalyKnowledgeBase":
        """
        Build from a list of historical case dicts.

        Each dict should contain:
            case_id     : str — unique identifier
            category    : str — transaction category
            description : str — free-text summary of the anomaly
            risk_factors: List[str] — feature names that triggered the flag
            resolution  : str — outcome / action taken
        """
        docs = []
        for rec in records:
            content = (
                f"Case ID: {rec.get('case_id', 'N/A')}\n"
                f"Category: {rec.get('category', 'N/A')}\n"
                f"Description: {rec.get('description', '')}\n"
                f"Risk factors: {', '.join(rec.get('risk_factors', []))}\n"
                f"Resolution: {rec.get('resolution', '')}"
            )
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "case_id":  rec.get("case_id", ""),
                        "category": rec.get("category", ""),
                    },
                )
            )
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        return self

    def build_from_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
    ) -> "AnomalyKnowledgeBase":
        """Build from plain-text documents (policy docs, runbooks, etc.)."""
        docs = [
            Document(page_content=t, metadata=m or {})
            for t, m in zip(texts, metadatas or [{}] * len(texts))
        ]
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        return self

    # -- Persistence

    def save(self, path: str | Path) -> None:
        if self.vectorstore is None:
            raise RuntimeError("Knowledge base is empty — nothing to save.")
        self.vectorstore.save_local(str(path))

    def load(self, path: str | Path) -> "AnomalyKnowledgeBase":
        self.vectorstore = FAISS.load_local(
            str(path),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        return self

    # -- Retrieval ----------------------------------------------------------

    def retrieve(self, query: str, k: int = None) -> str:
        """
        Return the top-k most similar historical cases as a single
        formatted string ready to inject into an LLM prompt.
        """
        k = k or config.RAG_TOP_K

        if self.vectorstore is None:
            return "No historical knowledge base available."

        results = self.vectorstore.similarity_search(query, k=k)
        if not results:
            return "No similar historical cases found."

        parts = [
            f"--- Similar Case {i} ---\n{doc.page_content}"
            for i, doc in enumerate(results, 1)
        ]
        return "\n\n".join(parts)
