# file: retrieval_parent.py

import time # <-- ADD THIS IMPORT
import asyncio
import numpy as np
import torch
import json
from groq import AsyncGroq
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from sklearn.preprocessing import MinMaxScaler
from torch.nn.functional import cosine_similarity
from typing import List, Dict, Tuple
from langchain.storage import InMemoryStore

from embedding import EmbeddingClient
from langchain_core.documents import Document

# --- Configuration ---
GENERATION_MODEL = "llama-3.1-8b-instant"
RERANKER_MODEL = 'cross-encoder/stsb-distilroberta-base'
INITIAL_K_CANDIDATES = 20
TOP_K_CHUNKS = 10 

async def generate_hypothetical_document(query: str, groq_api_key: str) -> str:
    """Generates a hypothetical document to answer the query (HyDE)."""
    if not groq_api_key:
        print("Groq API key not set. Skipping HyDE generation.")
        return ""

    print(f"Starting HyDE generation for query: '{query}'...")
    client = AsyncGroq(api_key=groq_api_key)
    prompt = (
        f"Write a brief, formal passage that directly answers the following question. "
        f"This passage will be used to find similar documents. "
        f"Do not include the question or any conversational text.\n\n"
        f"Question: {query}\n\n"
        f"Hypothetical Passage:"
    )
    
    start_time = time.perf_counter() # <-- START TIMER
    try:
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GENERATION_MODEL,
            temperature=0.7,
            max_tokens=500,
        )
        end_time = time.perf_counter() # <-- END TIMER
        print(f"--- HyDE generation took {end_time - start_time:.4f} seconds ---") # <-- PRINT DURATION
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"An error occurred during HyDE generation: {e}")
        return ""

async def generate_expanded_terms(query: str, groq_api_key: str) -> List[str]:
    """Generates semantically related search terms for a query."""
    if not groq_api_key:
        print("Groq API key not set. Skipping Semantic Expansion.")
        return [query] 

    print(f"Starting Semantic Expansion for query: '{query}'...")
    client = AsyncGroq(api_key=groq_api_key)
    prompt = (
        f"You are a search query expansion expert. Based on the following query, generate up to 4 additional, "
        f"semantically related search terms. The terms should be relevant for finding information in technical documents. "
        f"Return the original query plus the new terms as a single JSON list of strings.\n\n"
        f"Query: \"{query}\"\n\n"
        f"JSON List:"
    )

    start_time = time.perf_counter() # <-- START TIMER
    try:
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GENERATION_MODEL,
            temperature=0.4,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        end_time = time.perf_counter() # <-- END TIMER
        print(f"--- Semantic Expansion took {end_time - start_time:.4f} seconds ---") # <-- PRINT DURATION
        
        result_text = chat_completion.choices[0].message.content
        terms = json.loads(result_text)
        
        if isinstance(terms, dict) and 'terms' in terms:
            return terms['terms']
        return terms
    except Exception as e:
        print(f"An error occurred during Semantic Expansion: {e}")
        return [query]


class Retriever:
    """Manages hybrid search with parent-child retrieval."""

    def __init__(self, embedding_client: EmbeddingClient):
        self.embedding_client = embedding_client
        self.reranker = CrossEncoder(RERANKER_MODEL, device=self.embedding_client.device)
        self.bm25 = None
        self.document_chunks = []
        self.chunk_embeddings = None
        self.docstore = InMemoryStore()
        print(f"Retriever initialized with reranker '{RERANKER_MODEL}'.")

    def index(self, child_documents: List[Document], docstore: InMemoryStore):
        """Builds the search index from child documents and stores parent documents."""
        self.document_chunks = child_documents
        self.docstore = docstore
        
        corpus = [doc.page_content for doc in child_documents]
        if not corpus:
            print("No documents to index.")
            return

        print("Indexing child documents for retrieval...")
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.chunk_embeddings = self.embedding_client.create_embeddings(corpus)
        print("Indexing complete.")

    def _hybrid_search(self, query: str, hyde_doc: str, expanded_terms: List[str]) -> List[Tuple[int, float]]:
        """Performs a hybrid search using expanded terms for BM25 and a HyDE doc for dense search."""
        if self.bm25 is None or self.chunk_embeddings is None:
            raise ValueError("Retriever has not been indexed. Call index() first.")

        print(f"Running BM25 with expanded terms: {expanded_terms}")
        bm25_scores = self.bm25.get_scores(expanded_terms)

        enhanced_query = f"{query}\n\n{hyde_doc}" if hyde_doc else query
        query_embedding = self.embedding_client.create_embeddings([enhanced_query])
        dense_scores = cosine_similarity(query_embedding, self.chunk_embeddings).cpu().numpy().flatten()

        scaler = MinMaxScaler()
        norm_bm25 = scaler.fit_transform(bm25_scores.reshape(-1, 1)).flatten()
        norm_dense = scaler.fit_transform(dense_scores.reshape(-1, 1)).flatten()
        combined_scores = 0.5 * norm_bm25 + 0.5 * norm_dense
        
        top_indices = np.argsort(combined_scores)[::-1][:INITIAL_K_CANDIDATES]
        return [(idx, combined_scores[idx]) for idx in top_indices]

    async def _rerank(self, query: str, candidates: List[dict]) -> List[dict]:
        """Reranks candidates using a CrossEncoder model."""
        if not candidates:
            return []

        print(f"Reranking {len(candidates)} candidates...")
        rerank_input = [[query, chunk["content"]] for chunk in candidates]
        
        rerank_scores = await asyncio.to_thread(
            self.reranker.predict, rerank_input, show_progress_bar=False
        )

        for candidate, score in zip(candidates, rerank_scores):
            candidate['rerank_score'] = score
        
        candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
        return candidates[:TOP_K_CHUNKS]

    async def retrieve(self, query: str, groq_api_key: str) -> List[Dict]:
        """Executes the full retrieval pipeline: expansion, HyDE, hybrid search, and reranking."""
        print(f"\n--- Retrieving documents for query: '{query}' ---")

        hyde_task = generate_hypothetical_document(query, groq_api_key)
        expansion_task = generate_expanded_terms(query, groq_api_key)
        hyde_doc, expanded_terms = await asyncio.gather(hyde_task, expansion_task)

        initial_candidates_info = self._hybrid_search(query, hyde_doc, expanded_terms)
        
        retrieved_child_docs = [{
            "content": self.document_chunks[idx].page_content,
            "metadata": self.document_chunks[idx].metadata,
        } for idx, score in initial_candidates_info]

        reranked_child_docs = await self._rerank(query, retrieved_child_docs)

        parent_ids = []
        for doc in reranked_child_docs:
            parent_id = doc["metadata"]["parent_id"]
            if parent_id not in parent_ids:
                parent_ids.append(parent_id)

        retrieved_parents = self.docstore.mget(parent_ids)
        final_parent_docs = [doc for doc in retrieved_parents if doc is not None]

        final_context = [{
            "content": doc.page_content,
            "metadata": doc.metadata
        } for doc in final_parent_docs]

        print(f"Retrieved {len(final_context)} final parent chunks for context.")
        return final_context