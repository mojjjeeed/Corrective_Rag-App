# 🔍 RAGcraft — Progressive RAG Pipelines with LangGraph

A hands-on notebook series that builds a production-ready Retrieval-Augmented Generation (RAG) system from the ground up. Each notebook introduces one new concept on top of the last, so you can clearly see *why* each improvement matters.

Built with **LangChain**, **LangGraph**, **FAISS**, and **OpenAI**.

---

## 📚 Notebook Series

| # | Notebook | What It Adds |
|---|----------|--------------|
| 1 | `1_basic_rag.ipynb` | Baseline RAG — chunk PDFs, embed with FAISS, retrieve + generate |
| 2 | `2_retrieval_refinement.ipynb` | Sentence-level decomposition — filter noisy chunks before generating |
| 3 | `3_retrieval_evaluator.ipynb` | LLM-based doc scorer — routes to `fail` if no good docs are found |
| 4 | `4_web_search_refinement.ipynb` | Web search fallback — fetches live results when the vector store falls short |
| 5 | `5_query_rewrite.ipynb` | Query rewriting — rewrites the user question into a better web search query |
| 6 | `6_ambiguous.ipynb` | Ambiguity detection — catches vague questions before wasting retrieval effort |

---

## 🏗️ Architecture

Each notebook is a LangGraph `StateGraph` that evolves as the series progresses. By the final notebook, the full pipeline looks like this:

```
User Question
     │
     ▼
[Ambiguity Check] ──► Ambiguous? → Return clarification request
     │
     ▼
[Vector Store Retrieval]
     │
     ▼
[LLM Doc Evaluator]  (scores each doc 0.0 – 1.0)
     │
  ┌──┴──────────────┐
PASS (≥0.7)      FAIL (≤0.3)
  │                  │
  │            [Query Rewrite]
  │                  │
  │           [Web Search Fallback]
  │                  │
  └────────┬─────────┘
           ▼
   [Sentence Decomposer]
   (strip & filter retrieved text)
           │
           ▼
      [Generate Answer]
```

---

## ⚙️ Setup

**Prerequisites:** Python 3.10+, an OpenAI API key, and optionally a Tavily API key for web search.

```bash
pip install langchain langchain-community langchain-openai langgraph faiss-cpu pypdf pydantic tavily-python
```

Set your API keys:

```bash
export OPENAI_API_KEY="sk-..."
export TAVILY_API_KEY="tvly-..."   # needed for notebooks 4–6
```

Place your source PDFs in a `./documents/` folder:

```
documents/
  book1.pdf
  book2.pdf
  book3.pdf
```

---

## 🚀 Quickstart

Open the notebooks in order. Each one is self-contained and runnable top-to-bottom:

```bash
jupyter notebook 1_basic_rag.ipynb
```

Or run them all sequentially in JupyterLab.

---

## 🔑 Key Concepts Covered

- **Chunking & Embedding** — `RecursiveCharacterTextSplitter` + `text-embedding-3-large` + FAISS
- **Retrieval refinement** — sentence-level decomposition to keep only relevant sentences
- **LLM-as-judge** — structured scoring with Pydantic to evaluate retrieved docs
- **Conditional routing** — LangGraph edges that branch on retrieval quality
- **Web search fallback** — Tavily integration when local knowledge is insufficient
- **Query rewriting** — LLM rewrites vague user queries into precise search queries
- **Ambiguity detection** — early exit for questions that are too vague to answer reliably

---

## 🛠️ Tech Stack

- [LangChain](https://python.langchain.com/) — document loading, splitting, chains
- [LangGraph](https://langchain-ai.github.io/langgraph/) — stateful multi-step pipelines
- [FAISS](https://github.com/facebookresearch/faiss) — fast vector similarity search
- [OpenAI](https://platform.openai.com/) — `gpt-4o-mini` for generation, `text-embedding-3-large` for embeddings
- [Tavily](https://tavily.com/) — real-time web search API

---

## 📝 License

MIT
