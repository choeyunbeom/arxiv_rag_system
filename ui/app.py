"""
arXiv RAG System — Streamlit UI
Interactive Q&A interface over academic papers
"""

import streamlit as st
import httpx
import time

API_URL = "http://localhost:8000"

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
    .status-healthy { color: #4CAF50; }
    .status-degraded { color: #FF9800; }
</style>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.markdown("### Settings")
    top_k = st.slider("Number of sources", min_value=1, max_value=15, value=5)

    st.markdown("---")

    # Health check
    st.markdown("### System Status")
    try:
        health = httpx.get(f"{API_URL}/health", timeout=5.0).json()
        ollama_status = "Online" if health["ollama"] else "Offline"
        chroma_status = "Online" if health["chromadb"] else "Offline"

        st.markdown(f"**Ollama**: {ollama_status}")
        st.markdown(f"**ChromaDB**: {chroma_status}")
        st.markdown(f"**Indexed chunks**: {health['collection_count']:,}")
    except Exception:
        st.markdown("API server unreachable")
        st.markdown(f"Make sure the API is running at `{API_URL}`")

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
)

if st.button("🔍 Ask", type="primary", use_container_width=True) or (
    question and question != st.session_state.get("last_question", "")
):
    if not question:
        st.warning("Please enter a question.")
    else:
        st.session_state["last_question"] = question

        with st.spinner("Searching papers and generating answer..."):
            start = time.time()
            try:
                response = httpx.post(
                    f"{API_URL}/query",
                    json={"question": question, "top_k": top_k},
                    timeout=120.0,
                )
                response.raise_for_status()
                data = response.json()
                elapsed = time.time() - start

                # Answer
                st.markdown("### 💡 Answer")
                st.markdown(
                    f'<div class="answer-box">{data["answer"]}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(f"Generated in {elapsed:.1f}s")

                # Sources
                st.markdown("### Sources")
                for source in data["sources"]:
                    relevance = max(0, (1 - source["distance"]) * 100)
                    arxiv_url = f"https://arxiv.org/abs/{source['arxiv_id']}"

                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-title">
                            <a href="{arxiv_url}" target="_blank">{source['title']}</a>
                            <span class="distance-badge">{relevance:.0f}% relevant</span>
                        </div>
                        <div class="source-meta">
                            {source['authors']} · Section: {source['section']} · 
                            <a href="{arxiv_url}" target="_blank">arXiv:{source['arxiv_id']}</a>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            except httpx.ConnectError:
                st.error("Cannot connect to the API server. Make sure it's running on port 8000.")
            except Exception as e:
                st.error(f"Error: {e}")
