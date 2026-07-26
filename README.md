# 🔮 NexusRAG

**Multi-Document Agent with Hybrid Search, Web Augmentation & Knowledge Graph**

---

## ✨ What Makes This Special

| Feature | How |
|---|---|
| 🤖 **LLM** | Groq `llama-3.3-70b-versatile` — free tier (14,400 req/day) |
| 🔍 **Vector Search** | FAISS + `all-MiniLM-L6-v2` embeddings (runs locally) |
| 📊 **Sparse Search** | BM25 (rank_bm25) |
| ⚡ **Hybrid Fusion** | Reciprocal Rank Fusion (RRF) |
| 🌐 **Web Search** | DuckDuckGo (no API key, unlimited) |
| 🧠 **Knowledge Graph** | D3.js force graph, entity extraction from your docs |
| 📁 **File Types** | PDF, TXT, MD, DOCX, CSV |
| 🎯 **Smart Routing** | Agent decides: docs / web / hybrid per query |
| 🚀 **Deployment** | Streamlit Community Cloud (free forever) |

---

## 🛠️ Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/nexusrag
cd nexusrag

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py
```

## 🔑 Getting Your Free Groq Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (free, no card needed)
3. Click **API Keys** → **Create API Key**
4. Paste into NexusRAG sidebar

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│   Agent Router  │  ← decides: docs / web / hybrid
└────────┬────────┘
         │
    ┌────┴─────┐
    │          │
    ▼          ▼
┌──────┐  ┌─────────┐
│ FAISS│  │DuckDuck │
│+BM25 │  │Go Search│
│hybrid│  │(free)   │
└──┬───┘  └────┬────┘
   │           │
   └─────┬─────┘
         ▼
   ┌───────────┐
   │  Context  │  ← top-k chunks from FAISS+BM25 via RRF
   │  Builder  │  ← web snippets
   └─────┬─────┘
         ▼
   ┌───────────┐
   │  Groq LLM │  ← llama-3.3-70b (free)
   │  (free)   │
   └─────┬─────┘
         ▼
   Answer + Sources
```

---

## 📁 Project Structure

```
nexusrag/
├── app.py                    # Main Streamlit UI
├── requirements.txt
├── core/
│   ├── agent.py              # Main agent: routing + LLM calls
│   ├── document_processor.py # PDF/TXT/DOCX/CSV → semantic chunks
│   ├── vector_store.py       # FAISS + BM25 hybrid search
│   ├── web_search.py         # DuckDuckGo wrapper
│   └── entity_extractor.py   # NER for knowledge graph
└── utils/
    └── helpers.py            # UI formatting + D3 graph generator
```

---

## 🔮 Modes

| Mode | Behavior |
|---|---|
| **📄 Docs** | Only searches uploaded documents |
| **🌐 Web** | Only searches DuckDuckGo live web |
| **⚡ Hybrid** | Agent auto-routes based on query type |

---

## 💡 Example Queries

After uploading a research paper:
- *"Summarize the key contributions"*
- *"What methodology did they use?"*
- *"Compare this with BERT"* (hybrid: uses doc + web)
- *"What are the latest papers on this topic?"* (web only)

---

## 🌟 Tech Stack (All Free)

- **Streamlit** — UI framework (free cloud hosting)
- **Groq** — LLM inference (free tier)
- **sentence-transformers** — Local embeddings (MIT license)
- **FAISS** — Vector indexing (MIT license)
- **rank_bm25** — Sparse retrieval (Apache 2.0)
- **duckduckgo-search** — Web search (MIT, no key)
- **pypdf** — PDF parsing (MIT)
- **python-docx** — DOCX parsing (MIT)


