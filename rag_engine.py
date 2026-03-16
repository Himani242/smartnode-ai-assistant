import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory="vector_db",
    embedding_function=embedding_model,
    collection_name="smartnode_docs"
)

retriever = vector_db.as_retriever(search_kwargs={"k":3})

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)

def ask_ai(question):

    docs = retriever.invoke(question)

    if not docs:
        return "No relevant information found."

    context = "\n\n".join([doc.page_content[:400] for doc in docs[:3]])

    prompt = f"""
You are a Smart Node technical assistant.

Answer the question based only on the context.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    return response.content
