# 🤖 AI Agents — Guides & Deep Dives

> A growing collection of practical articles, code examples, and diagrams on building production-grade AI Agent systems.
>
> **Author:** Anand Topu | **Last Updated:** April 5, 2026

---

## 📚 Article Index

| # | Article | Topics | Diagrams |
|---|---------|--------|----------|
| 01 | [Building Memory-Augmented AI Agents: A Complete Guide](BuildingMemoryAugmentedAIAgents.md) | Agent Architecture · Stateless vs Memory Agents · Conversational Memory · Semantic Cache · Working Memory · Episodic Memory · Semantic Memory · Procedural Memory · Memory Manager · Vector Databases · Python | 8 |

---

## 🗂️ Topics Covered

### 🧠 Agent Fundamentals
- What is an AI Agent? (Perceive → Reason → Act → Remember)
- Four pillars: Perception, Reasoning, Action, Memory
- Stateless agents and their failure modes

### 🗄️ Agent Memory
- **Conversational Memory** — session history with SQLite
- **Semantic Cache** — vector similarity to avoid redundant LLM calls
- **Working Memory** — ephemeral in-context scratchpad
- **Episodic Memory** — timestamped interaction logs
- **Semantic Memory** — vector knowledge base with ChromaDB
- **Procedural Memory** — tool call and decision audit trail
- **Memory Manager** — unified orchestration layer
- **Memory Core (DB)** — the database backbone

### 🔧 Code & Tools
- Python (`openai`, `chromadb`, `sqlite3`, `numpy`)
- OpenAI GPT-4o & `text-embedding-3-small`
- ChromaDB / pgvector for vector storage
- SQLite for episodic and procedural stores
- Redis for semantic caching at scale

---

## 🖼️ Diagrams

All architecture diagrams are stored as both **Mermaid source** (`.mmd`) and **rendered PNG** (`.png`) in the [`diagrams/`](diagrams/) folder.

| File | Description |
|------|-------------|
| [`01_agent_architecture.png`](diagrams/01_agent_architecture.png) | High-level agent loop (Perceive → Reason → Act → Remember) |
| [`02_stateless_failures.png`](diagrams/02_stateless_failures.png) | Failure modes of stateless agents |
| [`03_memory_architecture.png`](diagrams/03_memory_architecture.png) | Memory-augmented agent — Agent Core, Memory Layer, Tools |
| [`04_context_window.png`](diagrams/04_context_window.png) | Context window limitations and truncation |
| [`05_memory_taxonomy.png`](diagrams/05_memory_taxonomy.png) | Full memory taxonomy (Short-Term & Long-Term) |
| [`06_memory_manager.png`](diagrams/06_memory_manager.png) | Memory Manager orchestration flowchart |
| [`07_memory_core_db.png`](diagrams/07_memory_core_db.png) | Agent Memory Core — database architecture |
| [`08_end_to_end_flow.png`](diagrams/08_end_to_end_flow.png) | End-to-end agent request sequence diagram |

---

## 📁 Repository Structure

```
AIAgents/
│
├── README.md                              ← You are here
│
├── BuildingMemoryAugmentedAIAgents.md     ← Article 01
│
└── diagrams/                              ← All architecture diagrams
    ├── 01_agent_architecture.{mmd,png}
    ├── 02_stateless_failures.{mmd,png}
    ├── 03_memory_architecture.{mmd,png}
    ├── 04_context_window.{mmd,png}
    ├── 05_memory_taxonomy.{mmd,png}
    ├── 06_memory_manager.{mmd,png}
    ├── 07_memory_core_db.{mmd,png}
    └── 08_end_to_end_flow.{mmd,png}
```

---

## 🔗 Further Reading

- [LangChain Memory Docs](https://python.langchain.com/docs/modules/memory/)
- [mem0 — Memory Layer for AI Agents](https://github.com/mem0ai/mem0)
- [ChromaDB — Open-source Vector Database](https://www.trychroma.com/)
- [pgvector — Vector Extension for PostgreSQL](https://github.com/pgvector/pgvector)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)

