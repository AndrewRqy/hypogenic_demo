# RAG-HypoGeniC (Modification of HypoGeniC)

This repository contains a lightweight modification of the original **HypoGeniC** implementation that adds an optional **Retrieval-Augmented Generation (RAG)** step to the hypothesis generation prompt.

The main entrypoint is `rag_hypogenic`, which runs the HypoGeniC pipeline while injecting retrieved literature chunks into the prompt (via `RAGPrompt`).

---

## What’s Added / Changed

### 1) `RAGPrompt` (hypogenic/rag_prompt)
A prompt wrapper that:
- builds the standard HypoGeniC generation prompt (unchanged),
- retrieves top-`k` relevant chunks from a vector store using a short retrieval query,
- injects the retrieved literature into the **user message** before generation.

Key parameters:
- `rag_enabled` (bool): turn RAG on/off
- `rag_k` (int): number of retrieved chunks
- `rag_max_chars` (int): maximum number of characters injected into the prompt

### 2) `rag_hypogenic(...)` (modification to the pipeline.py file)
A variant of the HypoGeniC runner that wires in:
- a `vectorstore` for retrieval,
- `RAGPrompt` for generation,
- standard HypoGeniC multi-hypothesis inference for evaluation (unless you changed this elsewhere).

### 3) `Retrieval Module` (rag.py)

Retrieval utilities module that:

- loads literature / document corpora,
- chunks documents into retrievable segments,
- builds embeddings for each chunk,
- creates and persists vector indices,
- loads existing vector indices for reuse.

Key components:
- `build_vectorstore(...)`: constructs a vector index from a corpus,
- chunking configuration (`chunk_size`, `chunk_overlap`),
- embedding backend configuration,
- persistent storage directory for saving / reloading indices.
---

##Example usage:
