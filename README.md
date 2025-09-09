# Document Intelligence Bajaj HackRx

This project is a high-performance, modular API for Retrieval-Augmented Generation (RAG) that leverages a sophisticated Parent-Child retrieval strategy. It is built with FastAPI and designed for efficient, parallel processing of complex documents to provide accurate answers to user questions.

## Table of Contents

  - [Key Features]
  - [Architecture Overview]
  - [Prerequisites]
  - [Installation]
  - [Configuration]
  - [Running the Application]
  - [API Endpoints]
      - [Run RAG Pipeline](https://percivalfletcher-chai-tea-latte.hf.space/hackrx/run)
      - [Test Ingestion Pipeline](https://percivalfletcher-chai-tea-latte.hf.space/test/ingestion)
      - [Test Chunking Strategy](https://percivalfletcher-chai-tea-latte.hf.space/test/chunk)
  - [Example Usage]

## Key Features

  - **Advanced Document Parsing**:
      - Parallel processing of PDF pages and ZIP archives for speed.
      - Utilizes **Table Transformer** and **EasyOCR** to accurately extract content from complex tables and images within documents.
      - Supports multiple file formats including PDF, DOCX, PPTX, JPG, PNG, and ZIP archives.
  - **Sophisticated Retrieval Strategy**:
      - Implements the **Parent-Child Chunking** strategy, using small, precise child chunks for searching and larger parent chunks for providing context to the LLM.
      - Employs **Hybrid Search** by combining dense (vector search) and sparse (BM25) retrieval methods for more robust results.
      - Enhances user queries with **HyDE** (Hypothetical Document Embeddings) and **Semantic Query Expansion** using an LLM.
      - Refines search results with a **Cross-Encoder Reranker** to ensure the most relevant context is passed to the generator.
  - **High-Performance Backend**:
      - Built with **FastAPI** for asynchronous, high-throughput performance.
      - Leverages the **Groq API** with Llama 3 for extremely fast final answer generation.
      - Modular design allows for easy extension and maintenance of individual components.

## Architecture Overview

The system follows a five-stage pipeline to process requests:

1.  **Ingestion & Parsing**: A document is downloaded from a URL and routed to the optimal parser based on its file type. Complex PDFs and ZIP files are processed in parallel to maximize efficiency.
2.  **Chunking**: The extracted text is processed using the Parent-Child strategy. Small child documents are created for retrieval, and larger parent documents are stored to provide context.
3.  **Indexing**: The child documents are indexed using two methods: dense vector embeddings for semantic search and a sparse BM25 index for keyword relevance.
4.  **Retrieval**: For each user question, the system enhances the query, performs a hybrid search on the indexes, and uses a reranker to find the most relevant child documents. It then fetches their corresponding parent documents to build the final context.
5.  **Generation**: The final context and the original question are sent to the Groq API, which generates a precise answer based solely on the provided information.

## Prerequisites

  - Python 3.8+
  - A virtual environment (recommended)

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Tanav-Kolar/Bajaj-HackRx6.0-ChadGPT.git
    cd Bajaj-HackRx6.0-ChadGPT
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    Create a `requirements.txt` file with the following contents:

    ```text
    fastapi
    pydantic
    python-dotenv
    uvicorn[standard]
    langchain
    langchain-core
    groq
    rank_bm25
    sentence-transformers
    scikit-learn
    torch
    transformers
    easyocr
    pandas
    Pillow
    unstructured
    python-pptx
    python-docx
    PyMuPDF
    httpx
    numpy
    ```

    Then, install the packages:

    ```bash
    pip install -r requirements.txt
    ```

## Configuration

The application requires API keys and other configuration to be stored in an environment file.

1.  Create a file named `.env` in the root directory of the project.

2.  Add the following required environment variable:

    ```env
    GROQ_API_KEY="your-groq-api-key"
    ```

## Running the Application

To start the API server, run the following command from the root directory:

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`. The interactive documentation (Swagger UI) can be accessed at `http://127.0.0.1:8000/docs`.

## API Endpoints

The API provides the following endpoints:

### Run RAG Pipeline

  - **Endpoint**: `POST /hackrx/run`
  - **Description**: The main endpoint that executes the entire RAG pipeline. It takes a document URL and a list of questions, then returns a list of generated answers.
  - **Request Body**:
    ```json
    {
      "documents": "https://example.com/document.pdf",
      "questions": [
        "What is the primary topic of the document?",
        "Summarize the key findings in section 2."
      ]
    }
    ```
  - **Response Body**:
    ```json
    {
      "answers": [
        "The primary topic is...",
        "Section 2 discusses..."
      ]
    }
    ```

### Test Ingestion Pipeline

  - **Endpoint**: `POST /test/ingestion`
  - **Description**: A testing endpoint to verify the document ingestion and parsing pipeline. It returns the extracted Markdown content and performance metrics.
  - **Request Body**:
    ```json
    {
      "documents": "https://example.com/document.pdf"
    }
    ```
  - **Response Body**:
    ```json
    {
      "total_time_seconds": 15.2,
      "character_count": 12050,
      "extracted_content": "# Document Title\n\nThis is the content..."
    }
    ```

### Test Chunking Strategy

  - **Endpoint**: `POST /test/chunk`
  - **Description**: A testing endpoint to verify the parent-child chunking strategy. It returns the generated parent and child chunks.
  - **Request Body**:
    ```json
    {
      "documents": "https://example.com/document.pdf"
    }
    ```
  - **Response Body**:
    ```json
    {
      "total_time_seconds": 16.1,
      "parent_chunk_count": 5,
      "child_chunk_count": 25,
      "parent_chunks": [
        {"page_content": "...", "metadata": {}}
      ],
      "child_chunks": [
        {"page_content": "...", "metadata": {"parent_id": "..."}}
      ]
    }
    ```

## Example Usage

You can interact with the main RAG pipeline using `curl`:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/hackrx/run' \
  -H 'Content-Type: application/json' \
  -d '{
    "documents": "https_link_to_your_pdf_or_docx_file",
    "questions": ["What is the capital of France?", "What is the conclusion mentioned in the document?"]
}'
```
