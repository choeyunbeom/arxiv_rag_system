"""
arXiv RAG System — Streamlit UI
Interactive Q&A interface over academic papers with error handling,
loading indicators, query history, and latency breakdown display.
"""

import os
import time

import httpx
import plotly.graph_objects as go
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

MAX_QUESTION_LENGTH = 500

# --- Page Config ---
st.set_page_config(
    page_title="arXiv RAG System",
    page_icon="📚",
    layout="wide",
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .source-card {
        background: #f8f9fa;
        border-left: 4px solid #4CAF50;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
    }
    .source-title {
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 4px;
    }
    .source-meta {
        font-size: 0.8rem;
        color: #888;
    }
    .distance-badge {
        display: inline-block;
        background: #e8f5e9;
        color: #2e7d32;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .answer-box {
        background: #fafafa;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
        line-height: 1.7;
    }
    .status-healthy { color: #4CAF50; font-weight: 600; }
    .status-degraded { color: #FF9800; font-weight: 600; }
    .status-offline { color: #f44336; font-weight: 600; }
    .latency-bar {
        display: flex;
        height: 8px;
        border-radius: 4px;
        overflow: hidden;
        margin: 4px 0 8px 0;
    }
    .latency-retrieval {
        background: #42A5F5;
    }
    .latency-generation {
        background: #FF7043;
    }
</style>
""", unsafe_allow_html=True)

# --- Session state init ---
if "query_history" not in st.session_state:
    st.session_state["query_history"] = []
if "last_question" not in st.session_state:
    st.session_state["last_question"] = ""


# --- Helper functions ---
def check_api_health():
    """Check API health and return status dict or None if unreachable."""
    try:
        health = httpx.get(f"{API_URL}/health", timeout=5.0).json()
        return health
    except Exception:
        return None


def render_latency_breakdown(latency: dict):
    """Render a visual latency breakdown bar and metrics."""
    retrieval = latency.get("retrieval_ms", 0)
    generation = latency.get("generation_ms", 0)
    total = latency.get("total_ms", 0)

    if total <= 0:
        return

    ret_pct = (retrieval / total) * 100
    gen_pct = (generation / total) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Retrieval", f"{retrieval / 1000:.1f}s")
    col2.metric("Generation", f"{generation / 1000:.1f}s")
    col3.metric("Total", f"{total / 1000:.1f}s")

    st.markdown(f"""
    <div class="latency-bar">
        <div class="latency-retrieval" style="width: {ret_pct:.1f}%;" title="Retrieval: {retrieval:.0f}ms"></div>
        <div class="latency-generation" style="width: {gen_pct:.1f}%;" title="Generation: {generation:.0f}ms"></div>
    </div>
    <span style="font-size: 0.75rem; color: #888;">
        🔵 Retrieval ({ret_pct:.0f}%) · 🟠 Generation ({gen_pct:.0f}%)
    </span>
    """, unsafe_allow_html=True)


def render_sources(sources: list):
    """Render source cards with safe field access."""
    st.markdown("### 📄 Sources")

    if not sources:
        st.info("No sources were retrieved for this query.")
        return

    for source in sources:
        title = source.get("title", "Untitled")
        arxiv_id = source.get("arxiv_id", "unknown")
        section = source.get("section", "N/A")
        authors = source.get("authors", "Unknown authors")
        distance = source.get("distance", 1.0)

        relevance = max(0, (1 - distance) * 100)
        arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"

        st.markdown(f"""
        <div class="source-card">
            <div class="source-title">
                <a href="{arxiv_url}" target="_blank">{title}</a>
                <span class="distance-badge">{relevance:.0f}% relevant</span>
            </div>
            <div class="source-meta">
                {authors} · Section: {section} ·
                <a href="{arxiv_url}" target="_blank">arXiv:{arxiv_id}</a>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_umap_visualization(vis_data: dict):
    """Render a 2D UMAP scatter plot using Plotly."""
    st.markdown("### 🌌 Embedding Space (UMAP)")
    if not vis_data or "points" not in vis_data:
        st.warning("No visualization data available.")
        return
    points = vis_data["points"]
    if not points:
        return

    fig = go.Figure()
    # Separate points
    bg_x, bg_y, bg_z, bg_text = [], [], [], []
    src_x, src_y, src_z, src_text = [], [], [], []
    query_x, query_y, query_z, query_text = [], [], [], []
    import textwrap
    for p in points:
        hover_text = f"<b>{p.get('title', 'Unknown')}</b><br>" \
                     f"Section: {p.get('section', 'N/A')}<br><br>" \
                     f"{textwrap.shorten(p.get('text_preview', ''), width=80)}"

        if p.get("is_query"):
            query_x.append(p["x"])
            query_y.append(p["y"])
            query_z.append(p.get("z", 0.0))
            query_text.append(hover_text)
        elif p.get("is_source"):
            src_x.append(p["x"])
            src_y.append(p["y"])
            src_z.append(p.get("z", 0.0))
            src_text.append(hover_text)
        else:
            bg_x.append(p["x"])
            bg_y.append(p["y"])
            bg_z.append(p.get("z", 0.0))
            bg_text.append(hover_text)

    # Background Points (Grey, small, faint)
    fig.add_trace(go.Scatter3d(
        x=bg_x, y=bg_y, z=bg_z,
        mode="markers",
        marker=dict(color="lightgrey", size=3, opacity=0.4),
        name="Papers",
        text=bg_text,
        hoverinfo="text"
    ))

    # Retrieved Sources (Green, medium, highlighted)
    fig.add_trace(go.Scatter3d(
        x=src_x, y=src_y, z=src_z,
        mode="markers",
        marker=dict(color="#4CAF50", size=8, line=dict(color="white", width=1), opacity=0.9),
        name="Retrieved Sources",
        text=src_text,
        hoverinfo="text"
    ))

    # User Query (Red star, large)
    fig.add_trace(go.Scatter3d(
        x=query_x, y=query_y, z=query_z,
        mode="markers",
        marker=dict(color="#F44336", symbol="diamond", size=10, line=dict(color="darkred", width=1)),
        name="Your Query",
        text=query_text,
        hoverinfo="text"
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        ),
        plot_bgcolor="white"
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


# --- Sidebar ---
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Number of sources", min_value=1, max_value=15, value=5)
    include_vis = st.toggle("🔍 Visualise Embeddings (UMAP)", value=True, help="Enable an interactive 3D map of the papers relative to your question. Requires pre-computed UMAP model.")

    st.markdown("---")

    # Health check with status badge
    st.markdown("### System Status")
    health = check_api_health()

    if health is None:
        st.markdown('<span class="status-offline">🔴 API Offline</span>', unsafe_allow_html=True)
        st.caption(f"Make sure the API is running at `{API_URL}`")
    else:
        overall = health.get("status", "unknown")
        if overall == "healthy":
            st.markdown('<span class="status-healthy">🟢 All Systems Healthy</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-degraded">🟡 Degraded</span>', unsafe_allow_html=True)

        ollama_icon = "✅" if health.get("ollama") else "❌"
        chroma_icon = "✅" if health.get("chromadb") else "❌"
        st.markdown(f"**Ollama**: {ollama_icon}")
        st.markdown(f"**ChromaDB**: {chroma_icon}")

        count = health.get("collection_count", 0)
        st.markdown(f"**Indexed chunks**: {count:,}")

    st.markdown("---")

    # Query history
    if st.session_state["query_history"]:
        st.markdown("### 🕐 Recent Queries")
        for i, q in enumerate(reversed(st.session_state["query_history"][-5:])):
            truncated = q[:60] + "..." if len(q) > 60 else q
            if st.button(truncated, key=f"hist_{i}", use_container_width=True):
                st.session_state["question"] = q

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown(
        "RAG system over **132 arXiv papers** on topics including "
        "RAG, QLoRA, LoRA, LLM fine-tuning, and hallucination mitigation."
    )
    st.markdown(
        "**Stack**: Qwen3 4B · mxbai-embed-large · ChromaDB · FastAPI"
    )


# --- Main ---
st.markdown('<p class="main-header">arXiv RAG System</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Ask questions about AI research papers — answers grounded in academic sources</p>',
    unsafe_allow_html=True,
)

# Example questions
examples = [
    "What is QLoRA and how does it reduce memory usage?",
    "How does Retrieval Augmented Generation work?",
    "What are common techniques to mitigate LLM hallucinations?",
    "How does LoRA differ from full fine-tuning?",
    "What evaluation metrics are used for RAG systems?",
]

st.markdown("**Try an example:**")
cols = st.columns(3)
for i, example in enumerate(examples):
    with cols[i % 3]:
        if st.button(example, key=f"ex_{i}", use_container_width=True):
            st.session_state["question"] = example

# Question input
question = st.text_input(
    "Your question",
    value=st.session_state.get("question", ""),
    placeholder="e.g., What is QLoRA and how does it work?",
    label_visibility="collapsed",
    max_chars=MAX_QUESTION_LENGTH,
)

# Input validation feedback
if question and len(question) > MAX_QUESTION_LENGTH * 0.9:
    st.caption(f"⚠️ {len(question)}/{MAX_QUESTION_LENGTH} characters")

# Action buttons
btn_col1, btn_col2 = st.columns([4, 1])
with btn_col1:
    ask_clicked = st.button("🔍 Ask", type="primary", use_container_width=True)
with btn_col2:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state["question"] = ""
        st.session_state["last_question"] = ""
        st.rerun()

# --- Query execution ---
should_query = ask_clicked or (
    question and question != st.session_state.get("last_question", "")
)

if should_query:
    if not question or not question.strip():
        st.warning("⚠️ Please enter a question.")
    elif len(question.strip()) < 3:
        st.warning("⚠️ Question is too short. Please enter at least 3 characters.")
    else:
        st.session_state["last_question"] = question

        # Add to history (avoid duplicates)
        history = st.session_state["query_history"]
        if not history or history[-1] != question:
            history.append(question)
            # Keep only last 10
            st.session_state["query_history"] = history[-10:]

        with st.status("Processing your question...", expanded=True) as status:
            st.write("🔍 Searching papers with hybrid retrieval...")
            start = time.time()

            try:
                if include_vis:
                    # Non-streaming path: needed for UMAP visualization data
                    response = httpx.post(
                        f"{API_URL}/query",
                        json={"question": question, "top_k": top_k, "include_vis": True},
                        timeout=120.0,
                    )
                    elapsed = time.time() - start

                    if response.status_code == 422:
                        status.update(label="Validation Error", state="error")
                        st.error("❌ Invalid input. Please check your question and try again.")
                    elif response.status_code == 503:
                        status.update(label="Service Unavailable", state="error")
                        st.error("🔌 **Ollama service is unavailable.**\n\nMake sure Ollama is running:\n```bash\nollama serve\n```")
                    elif response.status_code == 504:
                        status.update(label="Timeout", state="error")
                        st.error("⏱️ **LLM generation timed out.**\n\nTry a shorter or simpler question.")
                    elif response.status_code >= 400:
                        status.update(label="Error", state="error")
                        detail = response.json().get("detail", "Unknown error")
                        st.error(f"❌ Server error ({response.status_code}): {detail}")
                    else:
                        data = response.json()
                        status.update(label=f"Done in {elapsed:.1f}s", state="complete")

                        st.markdown("### 💡 Answer")
                        st.markdown(f'<div class="answer-box">{data["answer"]}</div>', unsafe_allow_html=True)

                        if "latency" in data and data["latency"]:
                            with st.expander("⏱️ Latency Breakdown", expanded=False):
                                render_latency_breakdown(data["latency"])

                        if data.get("vis_data"):
                            render_umap_visualization(data["vis_data"])

                        render_sources(data.get("sources", []))
                else:
                    # Streaming path: real-time token display via NDJSON
                    import json as _json

                    sources_data = []
                    latency_data = None
                    answer_text = ""

                    with httpx.stream(
                        "POST",
                        f"{API_URL}/query/stream",
                        json={"question": question, "top_k": top_k},
                        timeout=120.0,
                    ) as stream:
                        if stream.status_code >= 400:
                            status.update(label="Error", state="error")
                            error_body = ""
                            for chunk in stream.iter_text():
                                error_body += chunk
                            try:
                                detail = _json.loads(error_body).get("detail", error_body)
                            except _json.JSONDecodeError:
                                detail = error_body
                            st.error(f"❌ Server error ({stream.status_code}): {detail}")
                        else:
                            st.write("📊 Reranking and deduplicating results...")

                            # Prepare answer placeholder for streaming tokens
                            answer_header_shown = False
                            answer_placeholder = None

                            for line in stream.iter_lines():
                                if not line:
                                    continue
                                try:
                                    event = _json.loads(line)
                                except _json.JSONDecodeError:
                                    continue

                                evt_type = event.get("event")

                                if evt_type == "sources":
                                    sources_data = event.get("data", [])
                                    st.write("💡 Generating answer with Qwen3...")

                                elif evt_type == "token":
                                    if not answer_header_shown:
                                        status.update(label="Generating answer...", state="running")
                                        answer_header_shown = True
                                        # Close the status block and show answer area
                                        status.update(label="Streaming answer...", state="running")

                                    answer_text += event.get("data", "")

                                elif evt_type == "done":
                                    latency_data = event.get("data", {}).get("latency")

                                elif evt_type == "error":
                                    status.update(label="Error", state="error")
                                    st.error(f"❌ {event.get('data', 'Unknown error')}")

                            elapsed = time.time() - start
                            status.update(label=f"Done in {elapsed:.1f}s", state="complete")

                    # Render results after stream completes
                    if answer_text:
                        st.markdown("### 💡 Answer")
                        st.markdown(f'<div class="answer-box">{answer_text}</div>', unsafe_allow_html=True)

                    if latency_data:
                        with st.expander("⏱️ Latency Breakdown", expanded=False):
                            render_latency_breakdown(latency_data)

                    if sources_data:
                        render_sources(sources_data)

            except httpx.ConnectError:
                status.update(label="Connection Failed", state="error")
                st.error(
                    "🔌 **Cannot connect to the API server.**\n\n"
                    f"Make sure it's running at `{API_URL}`:\n"
                    "```bash\nuvicorn src.api.main:app --reload\n```"
                )
            except httpx.TimeoutException:
                status.update(label="Request Timeout", state="error")
                st.error(
                    "⏱️ **Request timed out** after 120 seconds.\n\n"
                    "The server may be overloaded. Try again or use a simpler question."
                )
            except httpx.HTTPStatusError as e:
                status.update(label="HTTP Error", state="error")
                st.error(f"❌ HTTP Error {e.response.status_code}: {e.response.text}")
            except Exception as e:
                status.update(label="Unexpected Error", state="error")
                st.error(
                    f"❌ **Unexpected error**: `{type(e).__name__}`\n\n"
                    f"```\n{str(e)}\n```\n\n"
                    "Please report this issue if it persists."
                )
