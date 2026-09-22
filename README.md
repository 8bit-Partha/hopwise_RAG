# Dev_RAG(hopwise) — Agentic Multi-Hop AI Engineering Assistant

An agent that answers AI/ML engineering questions by planning its own
research across three framework knowledge bases — LangChain, Hugging
Face, and vector databases — instead of running one fixed search. It
decides which knowledge base to check, examines what comes back,
decides whether it needs to check another one, and only answers once it
has enough. Every step is logged and shown, so the reasoning is
auditable, not just the final answer.

## Why this needs multiple hops

A single-hop RAG system answers with one search. That's fine for
"What does RunnableParallel do?" — but breaks down on questions real
engineers actually ask, like:

> *"If I want to use a Hugging Face AutoModel as an embedding model
> inside a LangChain VectorStoreRetriever backed by Chroma, what do I
> need to know about each piece?"*

No single knowledge base answers this — it spans all three frameworks.
DevRAG handles it by giving Claude three scoped tools and letting it
decide, turn by turn, where to look next:

```
search_langchain     — LCEL, Runnables, retrievers, chains, LangGraph
search_huggingface    — Transformers, pipeline(), PEFT/LoRA, the Hub
search_vectordb       — Chroma, Pinecone, Qdrant, Weaviate, HNSW
```

## Architecture

```
Query
  │
  ▼
┌────────────────────────────────────────────────┐
│  AGENT LOOP  (src/agent/agent_loop.py)            │
│  Claude (tool-use) reasons, optionally calls a    │
│  scoped tool, reads the result, decides: another  │
│  hop, or answer now? Hand-rolled on the Anthropic │
│  tool-use API — no framework — fully unit-tested  │
│  without a live API key.                          │
└────────────────────────────────────────────────┘
  │                    │                    │
  ▼                    ▼                    ▼
search_langchain   search_huggingface   search_vectordb
  │                    │                    │
  ▼                    ▼                    ▼
Each tool = its own Chroma collection + its own BM25 index,
built only from that domain's chunks, using the same
hybrid-search + cross-encoder re-rank pipeline throughout.
  │
  ▼
AgentTrace — every hop's reasoning, tool call, and result,
plus the final cited answer. Rendered in the Streamlit UI
and consumed directly by the eval harness.
```

## Project structure

```
devrag/
├── data/raw/                        # LangChain, HF, vector DB corpus
│   ├── langchain_concepts.md         # LCEL, Runnables, RAG chains, LangGraph
│   ├── langchain_components.json     # component reference (real API names)
│   ├── huggingface_concepts.md       # pipeline(), Auto classes, PEFT, Hub
│   ├── huggingface_components.json   # component reference
│   ├── vectordb_concepts.md          # ANN/HNSW, embedded vs client-server
│   └── vectordb_comparison.json      # Chroma/Pinecone/Qdrant/Weaviate/FAISS
├── src/
│   ├── ingestion/, chunking/, embeddings/, vectorstore/, retrieval/   (reused)
│   ├── tools/
│   │   ├── domains.py         # dependency-free source→domain mapping
│   │   └── retrieval_tools.py # scoped tool wrappers
│   ├── agent/
│   │   ├── prompts.py         # DevRAG system prompt
│   │   ├── trace.py           # structured reasoning trace
│   │   └── agent_loop.py      # the multi-hop control loop (DevRAGAgent)
│   └── pipeline.py            # build_index() + DevRAGPipeline
├── eval/
│   ├── multihop_test_questions.json  # 10 questions, single- and multi-hop
│   └── evaluate.py                    # agentic vs. single-hop baseline
├── app/streamlit_app.py      # chat UI showing the full reasoning trace
└── tests/                    # 30 tests, all runnable with no API key
```

## What I'd add next

- A fourth tool over real, larger documentation (e.g. actual LangChain
  API reference pages) once copyright-safe ingestion (linking + short
  fair-use excerpts rather than full reproduction) is worked out.
- Cache repeated searches within a single trace — if the agent
  re-queries the same tool with a near-duplicate query across hops,
  that's wasted latency worth catching.
- A "compare frameworks" mode that always fans out to all three tools in
  parallel rather than sequentially, for genuinely comparative
  questions.

## Tech stack

Python · sentence-transformers (BGE embeddings) · ChromaDB · rank-bm25 ·
Claude API (tool use) · Streamlit · Docker · pytest · GitHub Actions.

## License

MIT — see `LICENSE`.
