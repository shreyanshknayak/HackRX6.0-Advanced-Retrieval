# file: ingestion_router.py
import os
import time
import httpx
import zipfile
import io
import asyncio
from PIL import Image
from pathlib import Path
from urllib.parse import urlparse, unquote
from pydantic import HttpUrl
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Import our custom parsers ---
from pdf_parallel_parser import process_pdf_with_hybrid_parallel_sync
from complex_parser import process_image_element

# --- A simple parser for generic files (DOCX, etc.) using unstructured ---
from unstructured.partition.auto import partition

# --- Configuration ---
LOCAL_STORAGE_DIR = "data/"

# --- Synchronous, CPU-Bound Parsing Functions ---

def _process_generic_file_sync(file_content: bytes, filename: str) -> str:
    """Fallback parser for standard files like DOCX, PPTX, etc., using unstructured."""
    print(f"Processing '{filename}' with unstructured (standard)...")
    try:
        elements = partition(file=io.BytesIO(file_content), file_filename=filename)
        return "\n\n".join([el.text for el in elements])
    except Exception as e:
        print(f"Unstructured failed for {filename}: {e}")
        return ""

def _process_zip_file_in_parallel(zip_content: bytes, temp_dir: Path) -> str:
    """Extracts and processes files from a ZIP archive in parallel."""
    print("Initiating parallel processing of ZIP archive...")
    all_extracted_texts = []

    def process_single_zipped_file(zf: zipfile.ZipFile, file_info: zipfile.ZipInfo) -> str:
        file_content = zf.read(file_info.filename)
        file_extension = Path(file_info.filename).suffix.lower()
        
        # Route to the appropriate synchronous parser
        if file_extension == '.pdf':
            temp_file_path = temp_dir / Path(file_info.filename).name
            temp_file_path.parent.mkdir(parents=True, exist_ok=True)
            temp_file_path.write_bytes(file_content)
            return process_pdf_with_hybrid_parallel_sync(temp_file_path)
        elif file_extension in ['.png', '.jpg', '.jpeg']:
            return process_image_element(Image.open(io.BytesIO(file_content)))
        elif file_extension in ['.docx', '.pptx', '.html']:
            return _process_generic_file_sync(file_content, file_info.filename)
        else:
            print(f"Skipping unsupported file in ZIP: {file_info.filename}")
            return ""

    with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
        file_list = [info for info in zf.infolist() if not info.is_dir()]
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            future_to_file = {executor.submit(process_single_zipped_file, zf, file_info): file_info for file_info in file_list}
            for future in as_completed(future_to_file):
                try:
                    text = future.result()
                    if text:
                        all_extracted_texts.append(f"--- Content from: {future_to_file[future].filename} ---\n{text}")
                except Exception as e:
                    print(f"Error processing file '{future_to_file[future].filename}' from ZIP: {e}")
    
    return "\n\n".join(all_extracted_texts)

# --- Main Asynchronous Ingestion and Routing Function ---

async def ingest_and_parse_document(doc_url: HttpUrl) -> str:
    """Asynchronously downloads and parses a document using the optimal strategy."""
    print(f"Initiating processing for URL: {doc_url}")
    os.makedirs(LOCAL_STORAGE_DIR, exist_ok=True)
    start_time = time.perf_counter()

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(str(doc_url), timeout=120.0, follow_redirects=True)
            response.raise_for_status()
            doc_bytes = response.content
        print("Download successful.")

        filename = unquote(os.path.basename(urlparse(str(doc_url)).path)) or "downloaded_file"
        local_file_path = Path(LOCAL_STORAGE_DIR) / filename
        file_extension = local_file_path.suffix.lower()

        # Run the appropriate CPU-bound parsing function in a separate thread
        if file_extension == '.pdf':
            local_file_path.write_bytes(doc_bytes)
            doc_text = await asyncio.to_thread(process_pdf_with_hybrid_parallel_sync, local_file_path)
        elif file_extension == '.zip':
            doc_text = await asyncio.to_thread(_process_zip_file_in_parallel, doc_bytes, Path(LOCAL_STORAGE_DIR))
        elif file_extension in ['.png', '.jpg', '.jpeg']:
            image = Image.open(io.BytesIO(doc_bytes))
            doc_text = await asyncio.to_thread(process_image_element, image)
        elif file_extension in ['.docx', '.pptx', '.html']:
            doc_text = await asyncio.to_thread(_process_generic_file_sync, doc_bytes, filename)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        elapsed_time = time.perf_counter() - start_time
        print(f"Total processing time: {elapsed_time:.4f} seconds.")
        if not doc_text.strip():
            raise ValueError("Document parsing yielded no content.")

        return doc_text

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

# # Example of how to run the pipeline
# async def main():
#     # Example URL pointing to a PDF with tables
#     pdf_url = HttpUrl("https://www.w3.org/WAI/WCAG21/working-examples/pdf-table-linearized/table.pdf")
#     try:
#         content = await ingest_and_parse_document(pdf_url)
#         print("\n--- FINAL EXTRACTED CONTENT ---")
#         print(content)
#     except Exception as e:
#         print(f"Pipeline failed: {e}")

# if __name__ == "__main__":
#     asyncio.run(main())