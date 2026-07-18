import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Load the saved vector store (built by ingest.py)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer the user's question using only "
               "the context below. If the answer isn't in the context, say you don't know.\n\n"
               "Context:\n{context}"),
    ("placeholder", "{chat_history}"),
    ("human", "{question}"),
])

def ask(question: str, chat_history: list):
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "chat_history": chat_history,
        "question": question,
    })
    return response.content