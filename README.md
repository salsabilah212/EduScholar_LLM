# 🎓 EduScholarAI

AI assistant for education researchers and academics — search and summarization of theories from research catalog. Built with RAG (Retrieval-Augmented Generation) and LangChain.

---

## 📋 About

EduScholarAI is an AI assistant that answers education research questions based on provided documents, not internal model knowledge. It helps students, lecturers, and researchers find and summarize educational theories quickly and accurately with clear references.

**Key Features:**
- Search educational theories from research catalog
- Summarize concepts and research methodologies
- Provide answers with traceable sources
- Assist with literature review

---

## ⚙️ System Architecture

```
Research Catalog (TXT)
       ↓
  Document Loader
       ↓
  Text Splitter
       ↓
HuggingFace Embeddings
       ↓
  FAISS Vector Store
       ↓
    Retriever
       ↓
 LLM (Gemini/Groq/OpenAI)
       ↓
  Answer + Sources
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Gemini API / Groq (Llama 3.3) / OpenAI |
| RAG Framework | LangChain |
| Embeddings | HuggingFace (paraphrase-multilingual-MiniLM-L12-v2) |
| Vector Store | FAISS (local, no server) |
| UI | Streamlit |
| Language | Python 3.10+ |

---

## 📚 Topics in Research Catalog

1. Constructivism Theory
2. Cognitivism Theory
3. Behaviorism Theory
4. Quantitative Research
5. Qualitative Research
6. Classroom Action Research
7. Character Education
8. Learning Motivation Theory
9. Learning Assessment
10. Educational Technology

---

## 🚀 Setup & Run

### 1. Clone Repository

```bash
git clone https://github.com/username/eduscholar-rag.git
cd eduscholar-rag
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Create .env File

Copy `.env.example` and rename to `.env`:

```bash
cp .env.example .env
```

Add your API key (choose one):

```
GEMINI_API_KEY=AIzaSyxxxxxxxxxxxxxxxx
# or
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
# or
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
```

**Get API Key:**

| Provider | Steps |
|----------|-------|
| **Gemini** | https://ai.google.dev → Sign up → Create API Key |
| **Groq** | https://console.groq.com → Register → Create API Key |
| **OpenAI** | https://platform.openai.com → Sign up → Create API Key |

### 4. Run Application

```bash
streamlit run app.py
```

Open browser: http://localhost:8501

---

## 📁 Project Structure

```
eduscholar-rag/
├── app.py                      # Streamlit interface
├── rag_pipeline.py             # RAG logic with LangChain
├── requirements.txt            # Dependencies
├── .env.example                # API key template
├── .env                        # API key (DO NOT COMMIT)
├── .gitignore
├── README.md
└── data/
    └── eduresearchcatalog.txt  # Knowledge base
```

---

## ❓ Example Questions

- "What is the difference between quantitative and qualitative research?"
- "How to apply constructivism in the classroom?"
- "What are the theories of motivation in education?"
- "How to conduct Classroom Action Research?"
- "What is the difference between formative and summative assessment?"

---

## ☁️ Deployment to Streamlit Cloud

1. Push project to GitHub (make sure `.env` is not committed)

2. Go to https://share.streamlit.io

3. Connect your GitHub repository

4. Add Secrets in Streamlit dashboard:
   - Settings → Secrets
   - Add your API key:

   ```
   GEMINI_API_KEY = "AIzaSyxxxxxxxxxxxxxxxx"
   ```
   or
   ```
   GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxx"
   ```
   or
   ```
   OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxx"
   ```

5. Click Deploy

> **Note:** `.env` is not used on Streamlit Cloud. API keys are read from Secrets.

---

## 📝 Notes

- Research catalog contains 17 topics on educational theory and methodology
- Answers are based only on catalog documents, not general AI knowledge
- Sources are displayed for transparency and validation

---

## 📄 License

MIT License

## 👨‍💻 Contributor

[Your Name]