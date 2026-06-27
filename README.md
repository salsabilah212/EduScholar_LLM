# 🎓 EduScholar

> **AI-powered research assistant for educational research.**

EduScholar is a Retrieval-Augmented Generation (RAG) application that helps students, lecturers, and researchers explore educational theories and research methodologies from a trusted research repository.

Instead of relying solely on the Large Language Model (LLM), EduScholar retrieves the most relevant documents from its knowledge base before generating answers, ensuring responses are grounded, transparent, and supported by references.

---

## Features

- Explain educational theories
- Answer research methodology questions
- Maximum Marginal Relevance (MMR) retrieval
- Display source document previews
- Show author and publication year
- Modern chat interface built with Streamlit

---

## Preview



### Answer Generation


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

## RAG Workflow

EduScholar follows the Retrieval-Augmented Generation (RAG) workflow below.

```
                         User Question
                               │
                               ▼
                    Streamlit User Interface
                               │
                               ▼
                   Load Research Repository
                               │
                               ▼
                  Metadata Extraction
           (Title • Author • Publication Year)
                               │
                               ▼
            Recursive Character Text Splitter
          Chunk Size = 800 | Overlap = 100
                               │
                               ▼
          HuggingFace Embedding Model
(paraphrase-multilingual-MiniLM-L12-v2)
                               │
                               ▼
                 FAISS Vector Store
                               │
                               ▼
        MMR Retriever (Top-k Relevant Chunks)
      fetch_k = 8 | lambda = 0.7
                               │
                               ▼
          System Prompt + Retrieved Context
                               │
                               ▼
          Llama 3.3 70B (Groq API)
                               │
                               ▼
      Repository-based Answer + Sources
```

---

# Technology Stack

| Component | Technology |
|------------|------------|
| Programming Language | Python |
| User Interface | Streamlit |
| Framework | LangChain |
| LLM | Llama 3.3 70B (Groq) |
| Embedding Model | paraphrase-multilingual-MiniLM-L12-v2 |
| Vector Database | FAISS |
| Retrieval Strategy | Maximum Marginal Relevance (MMR) |
| Prompting | PromptTemplate |
| Chain | RetrievalQA |

---

# Knowledge Base

Current repository includes educational topics such as:

- Constructivism
- Cognitivism
- Behaviorism
- Learning Motivation
- Character Education
- Quantitative Research
- Qualitative Research
- Classroom Action Research
- Educational Assessment
- Educational Technology

---

# Example Questions

Examples of questions that EduScholar can answer:

- What is constructivism?
- Explain behaviorism and cognitivism.
- What is the difference between quantitative and qualitative research?
- How do I conduct Classroom Action Research?
- Explain multimedia learning principles.
- What are motivation theories in education?

---

# Project Structure

```
EduScholar
│
├── app.py
├── rag_pipeline.py
├── system_prompt.txt
├── requirements.txt
├── README.md
│
├── data
│   └── paper_literature.txt
│
└── .env
```

---

# Future Improvements
- PDF document ingestion
- Semantic search
- Research gap identification
- Multi-document comparison
- Research recommendation
- Cross-encoder reranker

---

# Author

**Salsabilah**

---

## License

MIT License