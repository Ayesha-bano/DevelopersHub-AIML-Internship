import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()  # loads GROQ_API_KEY from .env (not needed here, but good habit)

# Step A: Load every document from knowledge_base/
def load_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        if filename.endswith(".txt"):
            documents.extend(TextLoader(path, encoding="utf-8").load())
        elif filename.endswith(".pdf"):
            documents.extend(PyPDFLoader(path).load())
    return documents

documents = load_documents("./knowledge_base")
print(f"Loaded {len(documents)} document(s)")

# Step B: Split documents into smaller overlapping chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunk(s)")

# Step C: Convert chunks into vectors (embeddings) using a free local model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step D: Save everything into a local vector database
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
)
print("Vector store created and saved to ./chroma_db")