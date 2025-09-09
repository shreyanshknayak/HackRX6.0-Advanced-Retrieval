# file: main.py
import time
import os
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any
from dotenv import load_dotenv

# Assuming 'ingestion_router.py' is in the same directory and contains the function
from ingestion_router import ingest_and_parse_document
from chunking_parent import create_parent_child_chunks
from embedding import EmbeddingClient
from retrieval_parent import Retriever
from generation import generate_answer

load_dotenv()

app = FastAPI(
    title="Modular RAG API",
    description="A modular API for Retrieval-Augmented Generation with Parent-Child Retrieval.",
    version="2.3.0", # Updated version
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
embedding_client = EmbeddingClient()
retriever = Retriever(embedding_client=embedding_client)

# --- Pydantic Models ---
class RunRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]

class TestRequest(BaseModel):
    documents: HttpUrl

# --- NEW: Test Endpoint for Ingestion and Parsing ---
@app.post("/test/ingestion", response_model=Dict[str, Any], tags=["Testing"])
async def test_ingestion_endpoint(request: TestRequest):
    """
    Tests the complete ingestion and parsing pipeline.
    Downloads a document from a URL, processes it using the modular
    parsing strategy (e.g., parallel for PDF, standard for DOCX),
    and returns the extracted Markdown content and time taken.
    """
    print("--- Running Document Ingestion & Parsing Test ---")
    start_time = time.perf_counter()
    try:
        # Step 1: Call the main ingestion function from your router
        markdown_content = await ingest_and_parse_document(request.documents)

        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"--- Ingestion and Parsing took {duration:.2f} seconds ---")

        if not markdown_content:
            raise HTTPException(
                status_code=404,
                detail="Document processed, but no content was extracted."
            )

        return {
            "total_time_seconds": duration,
            "character_count": len(markdown_content),
            "extracted_content": markdown_content,
        }
    except Exception as e:
        # Catch potential download errors, parsing errors, or unsupported file types
        raise HTTPException(status_code=500, detail=f"An error occurred during ingestion test: {str(e)}")


# --- Test Endpoint for Parent-Child Chunking ---
@app.post("/test/chunk", response_model=Dict[str, Any], tags=["Testing"])
async def test_chunking_endpoint(request: TestRequest):
    """
    Tests the parent-child chunking strategy.
    Returns parent chunks, child chunks, and the time taken.
    """
    print("--- Running Parent-Child Chunking Test ---")
    start_time = time.perf_counter()

    try:
        # Step 1: Parse the document to get raw text
        markdown_content = await ingest_and_parse_document(request.documents)
        
        # Step 2: Create parent and child chunks
        child_documents, docstore, _ = create_parent_child_chunks(markdown_content)
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"--- Parsing and Chunking took {duration:.2f} seconds ---")

        # Convert Document objects to a JSON-serializable list for the response
        child_chunk_results = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in child_documents
        ]
        
        # Retrieve parent documents from the in-memory store
        parent_docs = docstore.mget(list(docstore.store.keys()))
        parent_chunk_results = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in parent_docs if doc
        ]
        
        return {
            "total_time_seconds": duration,
            "parent_chunk_count": len(parent_chunk_results),
            "child_chunk_count": len(child_chunk_results),
            "parent_chunks": parent_chunk_results,
            "child_chunks": child_chunk_results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during chunking test: {str(e)}")


@app.post("/hackrx/run", response_model=RunResponse)
async def run_rag_pipeline(request: RunRequest):
    try:
        print("--- Kicking off RAG Pipeline with Parent-Child Strategy ---")
        
        # --- STAGE 1: DOCUMENT INGESTION ---
        markdown_content = await ingest_and_parse_document(request.documents)
        
        # --- STAGE 2: PARENT-CHILD CHUNKING ---
        child_documents, docstore, _ = create_parent_child_chunks(markdown_content)

        if not child_documents:
            raise HTTPException(status_code=400, detail="Document could not be processed into chunks.")

        # --- STAGE 3: INDEXING ---
        retriever.index(child_documents, docstore)

        # --- STAGE 4: CONCURRENT RETRIEVAL & GENERATION ---
        print("Starting retrieval for all questions...")
        retrieval_tasks = [
            retriever.retrieve(q, GROQ_API_KEY)
            for q in request.questions
        ]
        all_retrieved_chunks = await asyncio.gather(*retrieval_tasks)
        print("Retrieval complete. Starting final answer generation...")

        answer_tasks = [
            generate_answer(q, chunks, GROQ_API_KEY)
            for q, chunks in zip(request.questions, all_retrieved_chunks)
        ]
        final_answers = await asyncio.gather(*answer_tasks)

        print("--- RAG Pipeline Completed Successfully ---")
        return RunResponse(answers=final_answers)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An internal server error occurred: {str(e)}"
        )
