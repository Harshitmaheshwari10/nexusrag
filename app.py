import streamlit as st
import pandas as pd 
import os
import json
import time
from pathlib import Path

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NexusRAG",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
  }

  /* Dark theme overrides */
  .stApp {
    background: #0a0a0f;
    color: #e8e8f0;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #0f0f1a !important;
    border-right: 1px solid #1e1e3a;
  }

  [data-testid="stSidebar"] * {
    color: #c8c8e0 !important;
  }

  /* Header */
  .nexus-header {
    background: linear-gradient(135deg, #0a0a0f 0%, #0d0d1f 50%, #0a0a0f 100%);
    border-bottom: 1px solid #1e1e3a;
    padding: 1.5rem 2rem;
    margin: -1rem -1rem 2rem -1rem;
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .nexus-logo {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #7c3aed, #3b82f6, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
  }

  .nexus-tagline {
    font-size: 0.8rem;
    color: #6b7280;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }

  /* Status badges */
  .status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
    font-family: 'JetBrains Mono', monospace;
  }

  .badge-web { background: #0d2a1a; color: #4ade80; border: 1px solid #166534; }
  .badge-docs { background: #1a0d2a; color: #c084fc; border: 1px solid #6d28d9; }
  .badge-hybrid { background: #0d1a2a; color: #60a5fa; border: 1px solid #1d4ed8; }

  /* Chat messages */
  .chat-bubble-user {
    background: #1a1a2e;
    border: 1px solid #2d2d4e;
    border-radius: 12px 12px 4px 12px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    max-width: 80%;
    margin-left: auto;
  }

  .chat-bubble-ai {
    background: #0f0f1f;
    border: 1px solid #1e1e3a;
    border-left: 3px solid #7c3aed;
    border-radius: 4px 12px 12px 12px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    max-width: 88%;
  }

  /* Source cards */
  .source-card {
    background: #0d0d1a;
    border: 1px solid #1e1e3a;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    margin: 0.3rem 0;
    font-size: 0.82rem;
    font-family: 'JetBrains Mono', monospace;
  }

  .source-card .score-bar {
    height: 3px;
    background: linear-gradient(90deg, #7c3aed, #3b82f6);
    border-radius: 2px;
    margin-top: 6px;
  }

  /* Doc cards in sidebar */
  .doc-card {
    background: #13131f;
    border: 1px solid #1e1e3a;
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    margin: 0.3rem 0;
    cursor: pointer;
    transition: all 0.15s;
  }

  .doc-card:hover { border-color: #7c3aed; }

  /* Agent mode selector */
  .mode-btn {
    padding: 8px 16px;
    border-radius: 8px;
    border: 1px solid #2d2d4e;
    background: #13131f;
    color: #c8c8e0;
    cursor: pointer;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.85rem;
    transition: all 0.15s;
  }

  .mode-btn.active {
    background: #1a0d2e;
    border-color: #7c3aed;
    color: #c084fc;
  }

  /* Metric cards */
  .metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.8rem;
    margin: 1rem 0;
  }

  .metric-card {
    background: #0f0f1a;
    border: 1px solid #1e1e3a;
    border-radius: 10px;
    padding: 0.8rem;
    text-align: center;
  }

  .metric-val {
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #7c3aed, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .metric-label {
    font-size: 0.72rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 2px;
  }

  /* Thinking indicator */
  .thinking-pulse {
    display: inline-flex;
    gap: 5px;
    align-items: center;
    padding: 6px 14px;
    background: #1a0d2e;
    border: 1px solid #6d28d9;
    border-radius: 20px;
    font-size: 0.78rem;
    color: #c084fc;
    font-family: 'JetBrains Mono', monospace;
  }

  /* Graph container */
  .graph-container {
    background: #07070f;
    border: 1px solid #1e1e3a;
    border-radius: 12px;
    padding: 1rem;
    min-height: 300px;
  }

  /* Stale button style for streamlit */
  .stButton > button {
    background: #13131f !important;
    color: #c8c8e0 !important;
    border: 1px solid #2d2d4e !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    transition: all 0.15s !important;
  }

  .stButton > button:hover {
    background: #1a0d2e !important;
    border-color: #7c3aed !important;
    color: #c084fc !important;
  }

  .stTextInput > div > div > input {
    background: #0f0f1a !important;
    border: 1px solid #2d2d4e !important;
    color: #e8e8f0 !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
  }

  .stTextInput > div > div > input:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 1px #7c3aed !important;
  }

  /* Hide streamlit default elements */
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  header { visibility: hidden; }

  /* Section headers */
  .section-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #6b7280;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.5rem;
  }

  /* Knowledge graph entity pill */
  .entity-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.72rem;
    margin: 2px;
    font-family: 'JetBrains Mono', monospace;
  }

  .entity-person { background: #1a0d2e; border: 1px solid #7c3aed; color: #c084fc; }
  .entity-org { background: #0d1a2e; border: 1px solid #2563eb; color: #60a5fa; }
  .entity-concept { background: #0d2a1a; border: 1px solid #166534; color: #4ade80; }
  .entity-date { background: #2a1a0d; border: 1px solid #92400e; color: #fbbf24; }

  /* Scroll area */
  .chat-scroll { max-height: 62vh; overflow-y: auto; padding-right: 8px; }
  .chat-scroll::-webkit-scrollbar { width: 4px; }
  .chat-scroll::-webkit-scrollbar-track { background: #0a0a0f; }
  .chat-scroll::-webkit-scrollbar-thumb { background: #2d2d4e; border-radius: 2px; }

  /* Web result card */
  .web-result {
    background: #0d1a0d;
    border: 1px solid #166534;
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    margin: 0.3rem 0;
    font-size: 0.82rem;
  }

  .web-result a { color: #4ade80; text-decoration: none; }
</style>
""", unsafe_allow_html=True)

# ─── Imports (lazy for speed) ────────────────────────────────────────────────
from core.document_processor import DocumentProcessor
from core.vector_store import VectorStoreManager
from core.agent import NexusAgent
from core.web_search import WebSearcher
from core.entity_extractor import EntityExtractor
from utils.helpers import format_sources, render_knowledge_graph_html

# ─── Session State Init ──────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents" not in st.session_state:
    st.session_state.documents = {}  # name -> metadata
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStoreManager()
if "agent" not in st.session_state:
    st.session_state.agent = None
if "mode" not in st.session_state:
    st.session_state.mode = "hybrid"  # docs | web | hybrid
if "entities" not in st.session_state:
    st.session_state.entities = {"persons": [], "orgs": [], "concepts": [], "dates": []}
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="nexus-logo" style="font-size:1.5rem; margin-bottom:0.2rem;">🔮 NexusRAG</div>', unsafe_allow_html=True)
    st.markdown('<div class="nexus-tagline">Multi-Doc Agent · Free · Open</div>', unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#1e1e3a; margin:1rem 0;'>", unsafe_allow_html=True)

    # API Key
    st.markdown('<div class="section-label">⚡ Groq API Key (Free)</div>', unsafe_allow_html=True)
    groq_key = st.text_input(
        "Groq Key",
        type="password",
        placeholder="gsk_...",
        label_visibility="collapsed",
        help="Get free key at console.groq.com — 14,400 req/day free"
    )
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        if st.session_state.agent is None or st.session_state.agent.groq_key != groq_key:
            st.session_state.agent = NexusAgent(
                groq_key=groq_key,
                vector_store=st.session_state.vector_store
            )
            st.success("✅ Agent ready!", icon="🤖")
    else:
        st.info("Add your free Groq key to enable AI responses", icon="🔑")
        st.markdown("""
        <div style="font-size:0.75rem; color:#6b7280; font-family:'JetBrains Mono',monospace; line-height:1.6;">
        1. Go to <b style="color:#c084fc">console.groq.com</b><br>
        2. Sign up free<br>
        3. Create API key<br>
        4. Paste above ↑
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1e1e3a; margin:1rem 0;'>", unsafe_allow_html=True)

    # Agent Mode
    st.markdown('<div class="section-label">🤖 Agent Mode</div>', unsafe_allow_html=True)
    mode_col1, mode_col2, mode_col3 = st.columns(3)
    with mode_col1:
        if st.button("📄 Docs", use_container_width=True,
                     type="primary" if st.session_state.mode == "docs" else "secondary"):
            st.session_state.mode = "docs"
    with mode_col2:
        if st.button("🌐 Web", use_container_width=True,
                     type="primary" if st.session_state.mode == "web" else "secondary"):
            st.session_state.mode = "web"
    with mode_col3:
        if st.button("⚡ Both", use_container_width=True,
                     type="primary" if st.session_state.mode == "hybrid" else "secondary"):
            st.session_state.mode = "hybrid"

    st.markdown(f"""
    <div style="font-size:0.72rem; color:#6b7280; font-family:'JetBrains Mono',monospace; margin-top:0.4rem; padding:6px 10px; background:#0d0d1a; border-radius:6px; border:1px solid #1e1e3a;">
    {'📄 Answers from your uploaded documents only' if st.session_state.mode == 'docs' else '🌐 Searches the web for real-time answers' if st.session_state.mode == 'web' else '⚡ Smart routing — docs + web, best of both'}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1e1e3a; margin:1rem 0;'>", unsafe_allow_html=True)

    # Document Upload
    st.markdown('<div class="section-label">📁 Upload Documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop files",
        accept_multiple_files=True,
        type=["pdf", "txt", "md", "docx", "csv"],
        label_visibility="collapsed"
    )

    if uploaded_files:
        for f in uploaded_files:
            if f.name not in st.session_state.documents:
                with st.spinner(f"Processing {f.name}..."):
                    try:
                        processor = DocumentProcessor()
                        chunks, metadata = processor.process(f)
                        chunk_ids = st.session_state.vector_store.add_documents(
                            chunks, metadata, doc_name=f.name
                        )
                        # Extract entities
                        extractor = EntityExtractor()
                        new_entities = extractor.extract_from_chunks(chunks)
                        for key in ["persons", "orgs", "concepts", "dates"]:
                            existing = set(st.session_state.entities[key])
                            existing.update(new_entities.get(key, []))
                            st.session_state.entities[key] = list(existing)[:20]

                        st.session_state.documents[f.name] = {
                            "chunks": len(chunks),
                            "size": f.size,
                            "type": f.type or "text/plain",
                        }
                        st.session_state.total_chunks += len(chunks)
                        st.success(f"✅ {f.name} — {len(chunks)} chunks", icon="📄")
                    except Exception as e:
                        st.error(f"Error: {e}")

    # Loaded docs list
    if st.session_state.documents:
        st.markdown("<hr style='border-color:#1e1e3a; margin:0.8rem 0;'>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">📚 Loaded Documents</div>', unsafe_allow_html=True)
        for doc_name, meta in st.session_state.documents.items():
            icon = "📄" if "pdf" in meta.get("type","") else "📝" if "text" in meta.get("type","") else "📊"
            st.markdown(f"""
            <div class="doc-card">
              <div style="font-weight:500; font-size:0.82rem; color:#e8e8f0;">{icon} {doc_name[:28]}{"…" if len(doc_name)>28 else ""}</div>
              <div style="font-size:0.7rem; color:#6b7280; font-family:'JetBrains Mono',monospace; margin-top:3px;">{meta['chunks']} chunks · {meta['size']//1024}KB</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1e1e3a; margin:0.8rem 0;'>", unsafe_allow_html=True)

    # Stats
    st.markdown('<div class="section-label">📊 Index Stats</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-card">
        <div class="metric-val">{len(st.session_state.documents)}</div>
        <div class="metric-label">Docs</div>
      </div>
      <div class="metric-card">
        <div class="metric-val">{st.session_state.total_chunks}</div>
        <div class="metric-label">Chunks</div>
      </div>
      <div class="metric-card">
        <div class="metric-val">{len(st.session_state.messages)//2}</div>
        <div class="metric-label">Turns</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Clear
    if st.button("🗑️ Clear Everything", use_container_width=True):
        st.session_state.messages = []
        st.session_state.documents = {}
        st.session_state.vector_store = VectorStoreManager()
        st.session_state.total_chunks = 0
        st.session_state.entities = {"persons": [], "orgs": [], "concepts": [], "dates": []}
        st.rerun()

# ─── Main Area ───────────────────────────────────────────────────────────────
col_main, col_graph = st.columns([3, 1.2])

with col_main:
    # Header
    st.markdown("""
    <div class="nexus-header">
      <div>
        <div class="nexus-logo">NexusRAG</div>
        <div class="nexus-tagline">Multi-Doc Agent · Hybrid Search · Knowledge Graph · 100% Free</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Mode indicator
    mode_map = {
        "docs": ('<span class="status-badge badge-docs">📄 DOCS MODE</span>', "Searching your uploaded documents"),
        "web": ('<span class="status-badge badge-web">🌐 WEB MODE</span>', "Searching the live web"),
        "hybrid": ('<span class="status-badge badge-hybrid">⚡ HYBRID MODE</span>', "Smart routing across docs + web"),
    }
    badge, desc = mode_map[st.session_state.mode]
    st.markdown(f"{badge} <span style='color:#6b7280; font-size:0.8rem; margin-left:8px;'>{desc}</span>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1rem;'>", unsafe_allow_html=True)

    # Chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-bubble-user">
                  <span style="font-size:0.72rem; color:#6b7280; font-family:'JetBrains Mono',monospace;">YOU</span>
                  <div style="margin-top:4px;">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                sources_html = ""
                if msg.get("sources"):
                    sources_html = format_sources(msg["sources"], msg.get("web_results", []))

                agent_icon = "🌐" if msg.get("source_type") == "web" else "📄" if msg.get("source_type") == "docs" else "⚡"
                st.markdown(f"""
                <div class="chat-bubble-ai">
                  <span style="font-size:0.72rem; color:#6b7280; font-family:'JetBrains Mono',monospace;">{agent_icon} NEXUS</span>
                  <div style="margin-top:6px; line-height:1.7;">{msg["content"]}</div>
                  {sources_html}
                </div>
                """, unsafe_allow_html=True)

    # Welcome screen
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center; padding:3rem 1rem; color:#6b7280;">
          <div style="font-size:3rem; margin-bottom:1rem;">🔮</div>
          <div style="font-size:1.1rem; font-weight:600; color:#c8c8e0; margin-bottom:0.5rem;">NexusRAG is ready</div>
          <div style="font-size:0.85rem; line-height:1.8; max-width:480px; margin:0 auto;">
            Upload PDFs, TXTs, or markdown files in the sidebar, then ask anything.<br>
            Switch modes to search docs only, web only, or let the agent decide.<br>
            <span style="color:#c084fc;">No subscriptions. No paid APIs. Completely free.</span>
          </div>
          <div style="margin-top:1.5rem; display:flex; gap:0.5rem; justify-content:center; flex-wrap:wrap;">
            <span style="padding:6px 14px; background:#0f0f1a; border:1px solid #1e1e3a; border-radius:20px; font-size:0.78rem;">"Summarize all documents"</span>
            <span style="padding:6px 14px; background:#0f0f1a; border:1px solid #1e1e3a; border-radius:20px; font-size:0.78rem;">"What are the key findings?"</span>
            <span style="padding:6px 14px; background:#0f0f1a; border:1px solid #1e1e3a; border-radius:20px; font-size:0.78rem;">"Compare doc A and B"</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Input
    st.markdown("<div style='margin-top:1rem;'>", unsafe_allow_html=True)
    input_col, btn_col = st.columns([5, 1])
    with input_col:
        user_input = st.text_input(
            "Ask anything",
            placeholder="Ask about your documents, or any topic...",
            label_visibility="collapsed",
            key="chat_input"
        )
    with btn_col:
        send_btn = st.button("Send →", use_container_width=True, type="primary")

    if (send_btn or user_input) and user_input.strip():
        if not st.session_state.agent:
            st.error("⚠️ Please add your Groq API key in the sidebar first.")
        else:
            query = user_input.strip()
            st.session_state.messages.append({"role": "user", "content": query})

            with st.spinner("🔮 Agent thinking..."):
                try:
                    response = st.session_state.agent.run(
                        query=query,
                        mode=st.session_state.mode,
                        chat_history=st.session_state.messages[:-1]
                    )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response.get("sources", []),
                        "web_results": response.get("web_results", []),
                        "source_type": response.get("source_type", "hybrid"),
                    })
                except Exception as e:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"⚠️ Error: {str(e)}",
                        "sources": [],
                        "web_results": [],
                        "source_type": "error",
                    })
            st.rerun()

# ─── Knowledge Graph Panel ────────────────────────────────────────────────────
with col_graph:
    st.markdown("""
    <div style="margin-top:4.5rem;">
      <div class="section-label" style="margin-bottom:0.8rem;">🧠 Knowledge Graph</div>
    </div>
    """, unsafe_allow_html=True)

    entities = st.session_state.entities
    has_entities = any(entities.values())

    if has_entities:
        graph_html = render_knowledge_graph_html(entities)
        st.components.v1.html(graph_html, height=320, scrolling=False)

        st.markdown("<div class='section-label' style='margin-top:1rem;'>Extracted Entities</div>", unsafe_allow_html=True)
        for cat, color_class in [("persons", "entity-person"), ("orgs", "entity-org"),
                                   ("concepts", "entity-concept"), ("dates", "entity-date")]:
            if entities[cat]:
                label = {"persons":"👤 People","orgs":"🏢 Orgs","concepts":"💡 Concepts","dates":"📅 Dates"}[cat]
                st.markdown(f"<div style='font-size:0.7rem; color:#6b7280; margin-top:0.5rem; margin-bottom:0.2rem;'>{label}</div>", unsafe_allow_html=True)
                pills = "".join([f'<span class="entity-pill {color_class}">{e[:18]}</span>' for e in entities[cat][:8]])
                st.markdown(pills, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="graph-container" style="display:flex; align-items:center; justify-content:center; flex-direction:column; gap:0.5rem;">
          <div style="font-size:2rem; opacity:0.3;">🕸️</div>
          <div style="font-size:0.78rem; color:#6b7280; text-align:center;">Upload documents to see entity graph</div>
        </div>
        """, unsafe_allow_html=True)

    # Query routing explainer
    st.markdown("<hr style='border-color:#1e1e3a; margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">🔀 Agent Routing Logic</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.72rem; line-height:1.9; color:#8888aa; font-family:'JetBrains Mono',monospace;">
    <span style="color:#c084fc;">if</span> query matches docs<br>
    &nbsp;&nbsp;<span style="color:#4ade80;">→ FAISS + BM25 hybrid</span><br>
    <span style="color:#c084fc;">elif</span> recency needed<br>
    &nbsp;&nbsp;<span style="color:#60a5fa;">→ DuckDuckGo search</span><br>
    <span style="color:#c084fc;">else</span><br>
    &nbsp;&nbsp;<span style="color:#fbbf24;">→ merge both + rerank</span><br>
    <br>
    <span style="color:#6b7280;">LLM: llama-3.3-70b</span><br>
    <span style="color:#6b7280;">Embed: MiniLM-L6-v2</span><br>
    <span style="color:#6b7280;">Search: DuckDuckGo</span><br>
    <span style="color:#6b7280;">Vector: FAISS</span>
    </div>
    """, unsafe_allow_html=True)
