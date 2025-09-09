# file: generation.py
from groq import AsyncGroq
from typing import List, Dict

# --- Configuration ---
GROQ_MODEL_NAME = "llama-3.1-8b-instant"

async def generate_answer(query: str, context_chunks: List[Dict], groq_api_key: str) -> str:
    """
    Generates a final answer using the Groq API based on the query and retrieved context.

    Args:
        query: The user's original question.
        context_chunks: A list of the most relevant, reranked document chunks.
        groq_api_key: The API key for the Groq service.

    Returns:
        A string containing the generated answer.
    """
    if not groq_api_key:
        return "Error: Groq API key is not set."
    if not context_chunks:
        return "I do not have enough information to answer this question based on the provided document."

    print("Generating final answer with Groq...")
    client = AsyncGroq(api_key=groq_api_key)

    # Format the context for the prompt
    context_str = "\n\n---\n\n".join(
        [f"Context Chunk:\n{chunk['content']}" for chunk in context_chunks]
    )

    prompt = (
        "You are an expert Q&A system. Your task is to extract information with 100% accuracy from the provided text. Provide a brief and direct answer."
        "Do not mention the context in your response. Answer *only* using the information from the provided document."
        "Do not infer, add, or assume any information that is not explicitly written in the source text. If the answer is not in the document, state that the information is not available."
        "When the question involves numbers, percentages, or monetary values, extract the exact figures from the text."
        "Double-check that the value corresponds to the correct plan or condition mentioned in the question."
        "Preserve the semantics of the language that the question has been asked in.\n\n"
        "\n\n"
        f"CONTEXT:\n{context_str}\n\n"
        f"QUESTION:\n{query}\n\n"
        "ANSWER:"
    )

    try:
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL_NAME,
            temperature=0.2, # Lower temperature for more factual answers
            max_tokens=500,
        )
        answer = chat_completion.choices[0].message.content
        print("Answer generated successfully.")
        return answer
    except Exception as e:
        print(f"An error occurred during Groq API call: {e}")
        return "Could not generate an answer due to an API error."