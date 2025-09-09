# file: complex_parser.py
import torch
import pandas as pd
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
import easyocr
from typing import List

# --- Configuration & Model Initialization ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Complex parser using device: {DEVICE}")

# Initialize models and reader once to save resources
TABLE_STRUCTURE_MODEL = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition").to(DEVICE)
IMAGE_PROCESSOR = AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
OCR_READER = easyocr.Reader(['en'])

# --- Helper Functions for Model Processing ---

def _get_bounding_box(tensor_box):
    """Converts a tensor bounding box to a PIL-compatible format."""
    return [round(i, 2) for i in tensor_box.tolist()]

def _get_cell_coordinates_by_row(table_data):
    """Organizes cell coordinates by their row."""
    rows = [sorted(row, key=lambda x: x['bbox'][0]) for row in table_data['rows']]
    return [{'row': i, 'bbox': _get_bounding_box(cell['bbox'])} for i, row in enumerate(rows) for cell in row]

def _apply_ocr_to_cells(image: Image.Image, cells: List[dict]) -> List[dict]:
    """Applies OCR to each cell in the table."""
    for cell in cells:
        cell_image = image.crop(cell['bbox'])
        ocr_result = OCR_READER.readtext(cell_image, detail=0, paragraph=True)
        cell['text'] = ' '.join(ocr_result)
    return cells

# --- Main Public Functions ---

def process_image_element(image: Image.Image) -> str:
    """Processes an image element using OCR to extract text."""
    print("--- Processing image element with OCR ---")
    try:
        # Convert the PIL Image to a NumPy array before passing to easyocr
        image_np = np.array(image)
        ocr_result = OCR_READER.readtext(image_np, detail=0, paragraph=True)
        text = ' '.join(ocr_result)
        return f"\n\n[Image Content: {text}]\n\n" if text else "\n\n[Image Content: No text detected]\n\n"
    except Exception as e:
        print(f"Error during image OCR: {e}")
        return "\n\n[Image Content: Error during processing]\n\n"

def process_table_element(image: Image.Image) -> str:
    """Processes a table element using Table Transformer and OCR."""
    print("--- Processing table element with Table Transformer ---")
    try:
        pixel_values, _ = IMAGE_PROCESSOR(image, return_tensors="pt")
        with torch.no_grad():
            outputs = TABLE_STRUCTURE_MODEL(pixel_values.to(DEVICE))
        
        table_data = outputs.to('cpu').item()
        if not table_data['rows']:
            return process_image_element(image)

        cells = _get_cell_coordinates_by_row(table_data)
        cells_with_text = _apply_ocr_to_cells(image, cells)
        
        df = pd.DataFrame(cells_with_text)
        if 'row' not in df.columns or 'text' not in df.columns:
            return "[Table Content: Could not form DataFrame]"
            
        table_pivot = df.pivot_table(index='row', columns=df.groupby('row').cumcount(), values='text', aggfunc='first').fillna('')
        markdown_table = table_pivot.to_markdown()
        
        return f"\n\n[Table Content]:\n{markdown_table}\n\n"
    except Exception as e:
        print(f"Error during table processing: {e}")
        return process_image_element(image)

def stitch_tables(table_markdowns: list[str]) -> str:
    """Stitches markdown tables from consecutive pages together."""
    if not table_markdowns:
        return ""
    full_table = table_markdowns[0]
    for i in range(1, len(table_markdowns)):
        lines = table_markdowns[i].split('\n')
        header_separator_index = next((j for j, line in enumerate(lines) if '|---' in line), -1)
        if header_separator_index != -1 and header_separator_index + 1 < len(lines):
            rows_to_append = '\n'.join(lines[header_separator_index + 1:])
            full_table += '\n' + rows_to_append
    return full_table