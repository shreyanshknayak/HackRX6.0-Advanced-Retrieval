# file: chunking.py
import uuid
from typing import List, Tuple, Dict, Any
from langchain_core.documents import Document
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configuration for Parent-Child Splitting ---
# Parent chunks are the larger documents passed to the LLM for context.
PARENT_CHUNK_SIZE = 2000
PARENT_CHUNK_OVERLAP = 200

# Child chunks are the smaller, more granular documents used for retrieval.
CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 100

def create_parent_child_chunks(
    full_text: str
) -> Tuple[List[Document], InMemoryStore, Dict[str, str]]:
    """
    Implements the Parent Document strategy for chunking.

    1. Splits the document into larger "parent" chunks.
    2. Splits the parent chunks into smaller "child" chunks.
    3. The child chunks are used for retrieval, while the parent chunks
       are used to provide context to the LLM.

    Args:
        full_text: The entire text content of the document.

    Returns:
        A tuple containing:
        - A list of the small "child" documents for the vector store.
        - An in-memory store mapping parent document IDs to the parent documents.
        - A dictionary mapping child document IDs to their parent's ID.
    """
    if not full_text:
        print("Warning: Input text for chunking is empty.")
        return [], InMemoryStore(), {}

    print("Creating parent and child chunks...")
    
    # This splitter creates the large documents that will be stored.
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
    )

    # This splitter creates the small, granular chunks for retrieval.
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
    )

    parent_documents = parent_splitter.create_documents([full_text])
    
    docstore = InMemoryStore()
    child_documents = []
    child_to_parent_id_map = {}

    # Generate unique IDs for each parent document and add them to the store
    parent_ids = [str(uuid.uuid4()) for _ in parent_documents]
    docstore.mset(list(zip(parent_ids, parent_documents)))

    # Split each parent document into smaller child documents
    for i, p_doc in enumerate(parent_documents):
        parent_id = parent_ids[i]
        _child_docs = child_splitter.split_documents([p_doc])
        
        for _child_doc in _child_docs:
            child_id = str(uuid.uuid4())
            _child_doc.metadata["parent_id"] = parent_id
            _child_doc.metadata["child_id"] = child_id
            child_to_parent_id_map[child_id] = parent_id
        
        child_documents.extend(_child_docs)

    print(f"Created {len(parent_documents)} parent chunks and {len(child_documents)} child chunks.")
    return child_documents, docstore, child_to_parent_id_map