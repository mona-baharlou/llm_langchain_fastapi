from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document


embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = None

def create_vectorstore(docs: list[Document]):
    global db
    db = FAISS.from_documents(docs, embedding)

def get_vectorstore() -> FAISS:
    return db
#db = None

