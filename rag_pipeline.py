import os
import re
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

# Configuration
DATA_PATH = "data/paper_literature.txt"
SYSTEM_PROMPT_PATH = "system_prompt.txt"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "llama-3.3-70b-versatile"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K_RESULTS = 4


def load_documents_with_metadata(file_path):
    """Load documents and extract metadata (title, author, year)"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    raw_docs = re.findall(r'--- DOCUMENT \d+ ---(.*?)(?=--- DOCUMENT \d+ ---|$)', content, re.DOTALL)
    documents = []
    
    for raw in raw_docs:
        if not raw.strip():
            continue
        
        lines = raw.strip().split('\n')
        
        # Extract metadata
        title = "Unknown"
        author = "Unknown"
        year = "Unknown"
        
        for line in lines[:5]:
            if line.startswith('TITLE:'):
                title = line.replace('TITLE:', '').strip()
            elif line.startswith('AUTHORS:') or line.startswith('AUTHOR:'):
                author = line.split(':', 1)[1].strip()
                if ':' in line:
                    author = line.split(':', 1)[1].strip()
            elif line.startswith('YEAR:'):
                year = line.replace('YEAR:', '').strip()
        
        # remove metadata lines
        clean_lines = []
        for line in lines:
            if not (line.startswith('TITLE:') or 
                    'AUTHORS' in line or 
                    'AUTHOR' in line or 
                    line.startswith('YEAR:') or
                    line.startswith('KEYWORDS')):
                clean_lines.append(line)
        
        clean_content = '\n'.join(clean_lines).strip()
        
        doc = Document(
            page_content=clean_content,
            metadata={
                'title': title,
                'author': author,
                'year': year,
            }
        )
        documents.append(doc)
    
    return documents


def load_system_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_rag_pipeline(top_k=TOP_K_RESULTS):
    # STEP 1: LOAD with metadata
    documents = load_documents_with_metadata(DATA_PATH)

    # STEP 2: CHUNK
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n---\n", "\n\n", "\n", " "]
    )
    chunks = splitter.split_documents(documents)

    # STEP 3: EMBEDDING
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # STEP 4: STORE
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # STEP 5: RETRIEVER
    #Notes: use MMR so that the retrieved documents are not only similar to the question but also diverse, thereby reducing redundant information.
    retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": top_k,
        "fetch_k": 8,
        "lambda_mult": 0.7
    }
)

    # STEP 6: LLM
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )

    # STEP 7: PROMPT
    prompt = PromptTemplate(
        template=load_system_prompt(SYSTEM_PROMPT_PATH),
        input_variables=["context", "question"]
    )

    # STEP 8: CHAIN
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return chain, len(chunks), documents 
