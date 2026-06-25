import streamlit as st
from rag_pipeline import build_rag_pipeline

# ── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduScholar",
    page_icon="🎓",
    layout="centered"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .chip-container { display: flex; flex-wrap: wrap; gap: 8px; margin: 0.5rem 0 1rem 0; }
    .chip {
        background: #EEF1FE; color: #4A6CF7;
        border: 1px solid #C7D2FE; border-radius: 20px;
        padding: 4px 12px; font-size: 0.82rem;
    }
    .sidebar-footer {
        text-align: center; font-size: 0.75rem;
        color: #9CA3AF; margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🎓 EduScholar")
st.caption("AI research assistant — find & summarize theories from education research repository")

# ── Load RAG Pipeline ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    return build_rag_pipeline()

if "pipeline_loaded" not in st.session_state:
    with st.status("⚙️ Loading AI system...", expanded=True) as status:
        st.write("📂 Reading research repository...")
        st.write("🔍 Building vector store...")
        st.write("🤖 Initializing LLM...")
        chain, num_chunks = load_pipeline()
        st.session_state.chain = chain
        st.session_state.num_chunks = num_chunks
        st.session_state.pipeline_loaded = True
        status.update(
            label=f"✅ System ready! {num_chunks} document chunks indexed.",
            state="complete"
        )

chain = st.session_state.chain

# ── Initialize Chat History ────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Example Questions (only when no chat yet) ──────────────────────────────
if not st.session_state.messages:
    st.info("💡 **Try asking:**")
    st.markdown("""
    <div class="chip-container">
        <span class="chip">📖 What is constructivism?</span>
        <span class="chip">🔬 Difference between quantitative & qualitative research</span>
        <span class="chip">🧠 Motivation theories in education</span>
        <span class="chip">📋 How to conduct Classroom Action Research</span>
        <span class="chip">💻 Multimedia learning principles</span>
        <span class="chip">⭐ How to apply behaviorism in the classroom?</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")  # spacing

# ── Display Chat History ───────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── User Input ──────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask about educational theory or research methodology..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching repository..."):
            result = chain.invoke({"query": user_input})
            answer = result["result"]
            source_docs = result["source_documents"]

        st.markdown(answer)

        with st.expander("📚 View Sources"):
            for i, doc in enumerate(source_docs, 1):
                st.markdown(f"**Source {i}:**")
                st.text(doc.page_content[:300] + "...")
                if i < len(source_docs):
                    st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 About EduScholar")
    st.markdown(
        "This application uses **RAG** *(Retrieval-Augmented Generation)* "
        "technology to answer questions based on the education research repository.\n\n"
        "> Answers are based **only** on repository documents, not general AI knowledge."
    )

    st.divider()

    st.markdown("### 📚 Topics in Catalog")
    st.markdown("""
| Category | Topics |
|---|---|
| Learning Theory | Constructivism, Cognitivism, Behaviorism |
| Methodology | Quantitative, Qualitative |
| Psychology | Learning Motivation, Character Education |
| Evaluation | Assessment, Educational Technology |
""")

    st.divider()

    st.markdown("### ⚙️ System Architecture")
    st.markdown(
        "```\n"
        "Research Catalog (TXT)\n"
        "       ↓\n"
        "  Document Loader\n"
        "       ↓\n"
        "  Text Splitter\n"
        "       ↓\n"
        "HuggingFace Embeddings\n"
        "       ↓\n"
        "  FAISS Vector Store\n"
        "       ↓\n"
        "    Retriever\n"
        "       ↓\n"
        " LLM (Gemini / Groq)\n"
        "       ↓\n"
        "  Final Answer\n"
        "```"
    )

    st.divider()

    if st.button("🔄 Reset Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        "<div class='sidebar-footer'>EduScholar · Powered by RAG + LLM</div>",
        unsafe_allow_html=True
    )