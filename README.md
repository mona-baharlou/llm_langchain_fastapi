# LLM PDF Chatbot with FastAPI + LangChain

This project implements a **chatbot backend** using [FastAPI](https://fastapi.tiangolo.com/) and [LangChain](https://www.langchain.com/) that can answer questions from a **PDF document** using a **Large Language Model (LLM)**.

> You upload a PDF, ask a question about its content, and the bot answers based on what's inside the PDF — not just general knowledge.


## Features

- Upload and process any PDF file
- Extract and chunk text using `PyPDFLoader`
- Embed PDF chunks with HuggingFace embeddings
- Store vector embeddings in FAISS for fast search
- Answer questions using a HuggingFace LLM (`google/flan-t5-base`)
- FastAPI backend with Swagger UI (`/docs`)

## Project Structure

![](https://github.com/mona-baharlou/llm_langchain_fastapi/blob/master/project_structure.PNG)

## How It Works
- PDF is split into small text chunks
- Each chunk is converted to vector using HuggingFace embeddings
- FAISS stores all vectors and does similarity search
- Most relevant chunks are passed to the LLM for response generation

## Tech Stack
 - LLM: google/flan-t5-base via HuggingFace Transformers
 - Embeddings: HuggingFace (all-MiniLM-L6-v2)
 - PDF parsing: PyMuPDF or PyPDFLoader
 - Backend: FastAPI
 - Vector DB: FAISS

## API Doc
![](https://github.com/mona-baharlou/llm_langchain_fastapi/blob/master/fastApi_Doc.PNG)

