# Import
import os
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY") 

from langchain_community.document_loaders import TextLoader            
from langchain_text_splitters import RecursiveCharacterTextSplitter     
from langchain_huggingface import HuggingFaceEmbeddings                 
from langchain_community.vectorstores import FAISS                     
from langchain_groq import ChatGroq                                    
from langchain.chains import RetrievalQA                               
from langchain.prompts import PromptTemplate  

load_dotenv()

# Configuration                          
# file location
DATA_PATH = "data/paper_literature.txt"
SYSTEM_PROMPT_PATH = "system_prompt.txt"

# Model
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "llama-3.3-70b-versatile"

# Chunk details
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

TOP_K_RESULTS = 4

# Load System Prompt
def load_system_prompt(path: str) -> str:
    """
    Reading the file system_prompt.txt and returning it as a string.
    The system prompt is stored in a separate file so that:
    - It is easy to modify without touching the Python code
    - It is more secure: system instructions are separated from program logic
    - It is cleaner: Python code focuses on logic, not long text
    
    The file uses XML-style delimiters for:
    - Structural clarity (each section has an opening and closing tag)
    - Security: LLMs are trained to respect XML tags as structural boundaries
    - Readability: anyone who opens the file immediately understands which part is what
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
    
SYSTEM_PROMPT_TEMPLATE = load_system_prompt(SYSTEM_PROMPT_PATH)

# Make a Function of Build Pipeline
def build_rag_pipeline():
    """
    Builds a complete RAG pipeline from scratch.

    Returns:
    - chain: a RetrievalQA object ready to accept queries
    - num_chunks: the number of text chunks successfully indexed
    """
    #-------------   
    #STEP 1: LOAD — Read the dataset file for the education paper
    loader = TextLoader(DATA_PATH, encoding="utf-8")
    documents = loader.load()

    # STEP 2: CHUNK — Breaking the document down into smaller sections
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
    
    # STEP 4: STORE — Saving the vector to FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # STEP 5: RETRIEVER — Setting up the search mechanism
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS}
    )
    
    # STEP 6: LLM — Initialize the language model using Groq
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    #STEP 7: PROMPT
    prompt = PromptTemplate(
        template=SYSTEM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    
    # STEP 8: CHAIN — Combining all components
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return chain, len(chunks)
