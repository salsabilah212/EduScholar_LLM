import streamlit as st
from rag_pipeline import build_rag_pipeline

# ── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduScholar",
    page_icon="🎓",
    layout="centered"
)

# Style
st.markdown("""
<style>
section[data-testid="stSidebar"] * { color: #e2e0f8 !important; }
h1 { color: #c4b5fd !important; }
.stButton > button {
    background: rgba(83,74,183,0.15) !important; color: #c4b5fd !important;
    border: 1px solid rgba(83,74,183,0.4) !important; border-radius: 12px !important;
    transition: all 0.2s !important;
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
        chain, num_chunks, documents = load_pipeline()
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

# ── Example Questions ──────────────────────────────────────────────────────
if "example_question" not in st.session_state:
    st.session_state.example_question = None
    
if not st.session_state.messages:
    st.markdown("### Suggested Questions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Constructivism", use_container_width=True):
            st.session_state.example_question = "What is constructivism?"
        if st.button("Motivation", use_container_width=True):
            st.session_state.example_question = "What are the theories of motivation in education?"
        if st.button("Classroom Action Research", use_container_width=True):
            st.session_state.example_question = "How to conduct Classroom Action Research?"

    with col2:
        if st.button("Quantitative vs Qualitative", use_container_width=True):
            st.session_state.example_question = "Difference between quantitative and qualitative research"
        if st.button("Multimedia Learning", use_container_width=True):
            st.session_state.example_question = "What are multimedia learning principles?"

# ── Display Chat History ───────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── User Input ──────────────────────────────────────────────────────────────
user_input = st.chat_input(
    "Ask about educational theory or research methodology..."
)
# If user use example question
if st.session_state.example_question:
    user_input = st.session_state.example_question
    st.session_state.example_question = None

if user_input:
    # Initialize variables outside try block
    answer = ""
    source_docs = []
    sources_were_used = False
    
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching repository..."):
            try:
                result = chain.invoke({"query": user_input})
                raw_answer = result["result"]
                source_docs = result["source_documents"]
                cleaned = raw_answer.strip()
                if cleaned.endswith("[SOURCES_USED]"):
                    sources_were_used = True
                    answer = cleaned[: -len("[SOURCES_USED]")].strip()
                elif cleaned.endswith("[NO_SOURCES]"):
                    sources_were_used = False
                    answer = cleaned[: -len("[NO_SOURCES]")].strip()
                else:
                    answer = cleaned
                    clarification_signals = [
                        "specify which", "please clarify", "which topic",
                        "could you clarify", "not available in the provided documents",
                        "i am an academic research assistant",  # model_extraction_guard reply
                    ]
                    lower_answer = answer.lower()
                    sources_were_used = not any(sig in lower_answer for sig in clarification_signals)
            except Exception as e:
                answer = "⚠️ Sorry, an error occurred while processing your question."
                source_docs = []
                st.error(str(e))

        st.markdown(answer)
        
        # ── VIEW SOURCES ──
        with st.expander("📚 View Sources"):
            # Filter dan hapus duplikat
            seen = set()
            unique_docs = []
            for doc in source_docs:
                title = doc.metadata.get('title', 'Unknown')
                # Hanya tampilkan jika judul tidak "Unknown" dan konten ada
                if title != 'Unknown' and title not in seen and len(doc.page_content) > 50:
                    seen.add(title)
                    unique_docs.append(doc)
            
            if unique_docs:
                st.write(f"**{len(unique_docs)} sources found:**")
                
                for i, doc in enumerate(unique_docs, 1):
                    title = doc.metadata.get('title', 'Unknown')
                    author = doc.metadata.get('author', 'Unknown')
                    year = doc.metadata.get('year', 'Unknown')
                    
                    # Tampilkan per source
                    with st.container(border=True):
                        st.markdown(f"**📘 {i}. {title}**")
                        st.caption(f"Author: {author} | Year: {year}")
                        st.text(doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content)
            else:
                st.info("No relevant sources found for this question.")
        
        # Append to messages AFTER all processing is complete
        st.session_state.messages.append({"role": "assistant", "content": answer})

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    with st.container(border=True):
        st.markdown("### 🎓 EduScholar")
        st.caption(
            "Your AI companion for educational research."
        )
        st.markdown("""
**Explore** learning theories

**Summarize** research literature

**Answer** using repository-based evidence
""")

    st.divider()

    st.markdown("### 📚 Topics")
    st.markdown("""
- Constructivism, Cognitivism, Behaviorism
- Quantitative & Qualitative Research
- Learning Motivation
- Character Education
- Assessment & Educational Technology
""")

    st.divider()

    if st.button("🔄 Reset Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.markdown("###  Powered by")
    st.markdown("""
- LangChain
- FAISS
- Groq
""")