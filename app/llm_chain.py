from app.vector_store import get_vectorstore
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from app.models import QueryResponse


# Load model once
model_id = "google/flan-t5-base"
pipe = pipeline("text2text-generation", model=model_id)
llm = HuggingFacePipeline(pipeline=pipe)

async def answer_question(question: str) -> QueryResponse:
    vectorstore = get_vectorstore()
    if vectorstore is None:
        return QueryResponse(answer="Please upload a PDF first.")

    retriever = vectorstore.as_retriever()
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = chain.run(question)
    return QueryResponse(answer=result)
