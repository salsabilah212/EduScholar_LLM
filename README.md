# 🎓 EduScholar

> **AI-powered Research Assistant for Educational Research**

EduScholar is a Retrieval-Augmented Generation (RAG) application that helps students, lecturers, and researchers explore educational theories and research methodologies through natural language interaction.

Unlike a conventional chatbot, EduScholar retrieves relevant information from an educational research repository before generating responses, ensuring that every answer is grounded in reliable sources rather than relying solely on the Large Language Model (LLM).

---

## Features

- Explain educational theories
- Answer research methodology questions
- Maximum Marginal Relevance (MMR) retrieval
- Display supporting source documents
- Show author and publication year
- Interactive chat interface using Streamlit

---

# System Architecture

```
                    User
                      │
                      ▼
             Streamlit Interface
                      │
                      ▼
               RetrievalQA Chain
                      │
      ┌───────────────┴───────────────┐
      ▼                               ▼
 FAISS Retriever                System Prompt
      │                               │
      ▼                               │
Educational Repository                │
      └───────────────┬───────────────┘
                      ▼
             Llama 3.3 (Groq API)
                      │
                      ▼
 Repository-based Answer + Sources
```

---

# RAG Pipeline

The system processes every question through eight stages:

| Step | Process | Description |
|------|----------|-------------|
| 1 | Document Loading | Load educational documents and extract metadata (title, author, publication year). |
| 2 | Text Chunking | Split documents into smaller chunks using Recursive Character Text Splitter. |
| 3 | Embedding | Convert every chunk into dense vector representations using HuggingFace MiniLM. |
| 4 | Vector Store | Store embeddings inside a FAISS vector database. |
| 5 | Retrieval | Retrieve relevant chunks using Maximum Marginal Relevance (MMR) to reduce redundant information. |
| 6 | LLM | Send retrieved context to Llama 3.3 through Groq API. |
| 7 | Prompting | Combine retrieved documents with a custom academic system prompt. |
| 8 | Answer Generation | Generate repository-based answers together with supporting source documents. |

---

## Technology Stack

| Component | Technology |
|------------|------------|
| Programming Language | Python |
| User Interface | Streamlit |
| Framework | LangChain |
| Large Language Model | Llama 3.3 (Groq API) |
| Embedding Model | paraphrase-multilingual-MiniLM-L12-v2 |
| Vector Database | FAISS |
| Retrieval Strategy | Maximum Marginal Relevance (MMR) |
| Prompt Engineering | PromptTemplate |
| QA Chain | RetrievalQA |

---

## Project Structure

```
EduScholar/
│
├── app.py
├── rag_pipeline.py
├── system_prompt.txt
├── requirements.txt
├── README.md
├── .env
│
├── data/
│   └── paper_literature.txt
│
└── images/
    ├── home.png
    ├── answer.png
    └── sources.png
```

---

# Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/EduScholar.git

cd EduScholar
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create an environment file

Create a `.env` file in the project root.

```text
GROQ_API_KEY=your_groq_api_key
```

---

# Run the Application

Run the Streamlit application.

```bash
streamlit run app.py
```

After the application starts, open your browser and visit:

```
http://localhost:8501
```

---

# Example Questions

You can ask questions such as:

- What is constructivism?
- Explain the theory of behaviorism.
- What is the difference between quantitative and qualitative research?
- How do I conduct Classroom Action Research?
- What are the principles of multimedia learning?
- Explain motivation theories in education.

---

# Future Improvements

- PDF document ingestion
- Citation generation (APA Style)
- Literature review synthesis
- Semantic search
- Multi-document comparison
- Research recommendation
- Research gap identification
- Cross-encoder reranker
- Conversation memory

---

# Author

**Salsabilah**

---

# 📄 License

This project is released under the MIT License.