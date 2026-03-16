import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


# Load API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector DB
vector_db = Chroma(
    persist_directory="vector_db",
    embedding_function=embedding_model,
    collection_name="smartnode_docs"
)

retriever = vector_db.as_retriever(search_kwargs={"k":5})

# Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)


def ask_ai(question):

    docs = retriever.invoke(question)

    if not docs:
        return "No relevant information found."

    context = "\n\n".join([doc.page_content[:800] for doc in docs])

    prompt = f"""
You are an assistant for Smart Node company.

Answer the question based only on the context.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return response.content
