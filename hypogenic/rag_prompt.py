from abc import ABC, abstractmethod
import os
import textwrap
from string import Template
from typing import List, Tuple, Union, Dict
from copy import deepcopy
import pandas as pd
import json
from .prompt import BasePrompt

from hypogenic.logger_config import LoggerConfig

logger = LoggerConfig.get_logger("Prompt")



class RAGPrompt(BasePrompt):
    def __init__(self, task, vectorstore, rag_k=4, rag_max_chars=3000, rag_enabled=True):
        super().__init__(task)
        self.vectorstore = vectorstore
        self.rag_k = rag_k
        self.rag_max_chars = rag_max_chars
        self.rag_enabled = rag_enabled

    def _retrieve(self, query: str) -> str:
        assert self.vectorstore is not None, "RAG enabled but vectorstore is None"
        #logger.info("[RAG] Query: %s", query[:200].replace("\n", " "))

        logger.info("[RAG] Retrieval query:\n%s", query.replace("\n", " "))
    
        results = self.vectorstore.similarity_search_with_score(query, k=self.rag_k)
        logger.info("[RAG] Retrieved %d chunks", len(results))

        # Also log the top few chunks right after retrieval
        for i, (doc, score) in enumerate(results[: self.rag_k], start=1):
            src  = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", None)
            logger.info(
                "[RAG] Chunk %d | score=%.4f | source=%s | page=%s\n%s",
                i,
                score,
                src,
                str(page),
                doc.page_content.strip()[:500],  # limit to first 500 chars
            )
        tops = []
        for doc, score in results[:3]:
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", None)
            tops.append(f"{src}:{page} score={score:.4f}")
        logger.info("[RAG] Top hits: %s", " | ".join(tops) if tops else "none")

        blocks, total = [], 0
        for rank, (doc, score) in enumerate(results, start=1):
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", None)
            header = (
                f"[Lit #{rank} | score={score:.4f} | source={src}"
                + (f" | page={page}]" if page is not None else "]")
            )
            chunk = f"{header}\n{doc.page_content.strip()}"

            if total + len(chunk) > self.rag_max_chars:
                break
            blocks.append(chunk)
            total += len(chunk)

        ctx = "\n\n".join(blocks) if blocks else "[No retrieved literature]"
        logger.info("[RAG] Injected chars: %d", len(ctx))
        return ctx

    def _inject_rag(self, prompt, query: str):
        """prompt is typically a list of messages: [system_msg, user_msg]"""
        logger.info("[RAG] Injecting literature into prompt")
        if not self.rag_enabled:
            return prompt

        logger.info("[RAG] Calling _retrieve now...")
        lit = self._retrieve(query)
        logger.info("[RAG] _retrieve returned %d chars", len(lit))

        # Add a marker so you can verify at runtime
        prompt[1]["content"] = (
            f"[RAG_ENABLED=true | k={self.rag_k}]\n"
            f"Relevant scientific literature (retrieved):\n{lit}\n\n"
            + prompt[1]["content"]
        )
        return prompt

    def batched_generation(self, example_bank, num_hypotheses):
        # 1) Build the original prompt exactly as BasePrompt would
        prompt = super().batched_generation(example_bank, num_hypotheses)

        # 2) Build a retrieval query based on the current task + a few examples
        # Keep it short to avoid noisy queries
        preview_rows = []
        for i in range(min(5, len(example_bank))):
            try:
                preview_rows.append(example_bank.iloc[i].to_dict())
            except Exception:
                preview_rows.append(str(example_bank.iloc[i]))

        query = (
            f"Task: {getattr(self.task, 'task_name', '')}. "
            f"Goal: generate interpretable hypotheses that distinguish labels. "
            f"Examples: {json.dumps(preview_rows, ensure_ascii=False)[:1200]}"
        )

        # 3) Inject retrieved literature into the user prompt
        prompt = self._inject_rag(prompt, query)
        return prompt