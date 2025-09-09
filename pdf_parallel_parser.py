# file: pdf_parallel_parser.py

import fitz  # PyMuPDF
from PIL import Image
import io
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Import the specialized parsers from our other module
from complex_parser import process_table_element, process_image_element

def _is_bbox_contained(inner_bbox, outer_bbox):
    """Check if inner_bbox is fully inside outer_bbox."""
    return (inner_bbox[0] >= outer_bbox[0] and
            inner_bbox[1] >= outer_bbox[1] and
            inner_bbox[2] <= outer_bbox[2] and
            inner_bbox[3] <= outer_bbox[3])

def _process_page(page: fitz.Page) -> str:
    """
    Processes a single PDF page to extract text, tables, and images.
    - Tables are found and processed with the complex_parser.
    - Plain text is extracted, excluding any text already inside a processed table.
    """
    page_content = []
    
    # 1. Find and process tables first
    table_bboxes = []
    try:
        tables = page.find_tables()
        pix = page.get_pixmap(dpi=200)
        page_image = Image.open(io.BytesIO(pix.tobytes("png")))
        
        print(f"Page {page.number}: Found {len(tables.tables)} potential tables.")
        for i, table in enumerate(tables):
            table_bboxes.append(table.bbox)
            table_image = page_image.crop(table.bbox)
            markdown_table = process_table_element(table_image)
            page_content.append(markdown_table)
    except Exception as e:
        print(f"Could not process tables on page {page.number}: {e}")

    # 2. Extract text blocks, excluding those within table bounding boxes
    text_blocks = page.get_text("blocks")
    for block in text_blocks:
        block_bbox = block[:4]
        # Check if this text block is inside any of the tables we just processed
        is_in_table = any(_is_bbox_contained(block_bbox, table_bbox) for table_bbox in table_bboxes)
        if not is_in_table:
            page_content.append(block[4].strip())
            
    # Note: Image extraction can be added here if needed, similar to table extraction.

    return "\n".join(page_content)

def process_pdf_with_hybrid_parallel_sync(file_path: Path) -> str:
    """
    Processes a PDF file in parallel using PyMuPDF and the complex_parser.
    """
    print(f"Processing PDF '{file_path.name}' with parallel page-by-page strategy...")
    all_page_texts = []
    doc = fitz.open(file_path)

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = {executor.submit(_process_page, page): page.number for page in doc}
        
        # Collect results in page order
        results = ["" for _ in range(len(doc))]
        for future in as_completed(futures):
            page_num = futures[future]
            try:
                results[page_num] = future.result()
            except Exception as e:
                print(f"Error processing page {page_num}: {e}")
        all_page_texts = results

    return f"\n\n--- Page Break ---\n\n".join(all_page_texts)