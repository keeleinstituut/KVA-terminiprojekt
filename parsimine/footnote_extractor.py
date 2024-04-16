from collections import defaultdict
import re
from typing import List
import fitz  # PyMuPDF
import json
from parser_utils import calculate_content_box

class FootnoteExtractor:
    """
    A class to extract footnotes from PDF documents using PyMuPDF (fitz).
    
    Args:
        - pattern (str): The regular expression pattern used to identify footnotes.
        - footnote_group (int): The group number in the regular expression to extract the footnote text.
    
    Methods:
        - extract_footnotes_from_page(page: fitz.Page, header_height: int =  0, footer_height: int =  0) -> str: Extracts footnotes from a single PDF page.
        - extract_footnotes_from_doc(doc: fitz.Document, header_height: int =  0, footer_height: int =  0) -> str: Extracts footnotes from an entire PDF document.
        - __is_footnote(text: str, current_block_index: int, total_blocks: int) -> bool: Checks if a text block is a footnote based on the pattern and position.
        - __extract_footnotes(text: str) -> List[str]: Splits a block of text into individual footnotes based on the pattern.
    """
        
    def __init__(self, footnote_pattern: str = r'^\d+\s+(.*)', footnote_group: int = 1):
        self.pattern = footnote_pattern
        self.footnote_group = footnote_group

    def extract_footnotes_from_page(self, page: fitz.Page, header_height: int = 0, footer_height: int= 0) -> str:
        """
        Extract footnotes from a single PDF page.
        
        Args:
            - page (fitz.Page): The page from which to extract footnotes.
            - header_height (int): The height of the page header to exclude from extraction.
            - footer_height (int): The height of the page footer to exclude from extraction.
        
        Returns:
            - str: A JSON string representing the extracted footnotes, or None if no footnotes were found.
        """
        footnotes = list()
        footnote_bboxes = list()

        page_blocks = page.get_text('blocks', sort=True, clip=calculate_content_box(page, header_height, footer_height))
        
        total_blocks_count = len(page_blocks)

        for index, block in enumerate(page_blocks):
            if self.__is_footnote(block[4], index, total_blocks_count):
                footnote_bboxes.append(block[0:4])
                footnotes.append(self.__extract_footnotes(block[4].strip())) 

        # Prepare the data for JSON serialization, including the page number.
        footnotes_json = [{
            "page_number": page.number + 1,  # Page numbers are zero-based in PyMuPDF
            "texts": footnotes,
            "bboxes": footnote_bboxes,
        }]

        if not footnotes:
            return None
        return json.dumps(footnotes_json)

    def extract_footnotes_from_doc(self, doc: fitz.Document, header_height: int = 0, footer_height: int = 0) -> str:
        """
        Extract footnotes from an entire PDF document.
        
        Args:
            - doc (fitz.Document): The PDF document from which to extract footnotes.
            - header_height (int): The height of the page header to exclude from extraction.
            - footer_height (int): The height of the page footer to exclude from extraction.
        
        Returns:
            - str: A JSON string representing all extracted footnotes from the document.
        """
        all_footnotes = []
        for page in doc:
            # Use the previously defined method to extract tables from each page
            footnotes_json = self.extract_footnotes_from_page(page, header_height=header_height, footer_height=footer_height)
            if footnotes_json:
                all_footnotes.extend(json.loads(footnotes_json))
        return json.dumps(all_footnotes)
    
    
    def __is_footnote(self, text: str, current_block_index: int, total_blocks: int) -> bool:
        """
        Conditions for a valid footenote:
        1. Footnote begins with a number and space combination, followed by a word. One block may contain multiple footenotes, thats why re.MULTILINE flag is raised.
        2. There are more blocks before than after a footnote.
        """
        footnote_pattern = self.pattern
        footnote_match = re.match(footnote_pattern, text.strip(), re.MULTILINE)
        
        is_pattern_footnote = bool(footnote_match)
        is_position_footnote = (total_blocks - current_block_index) < current_block_index

        if is_pattern_footnote and is_position_footnote:
            return True
        return False

    def __extract_footnotes(self, text: str) -> List[str]:
        """
        Splits a block of text into individual footnotes based on the pattern.
        
        Args:
            - text (str): The text block to split.
        
        Returns:
            - List[str]: A list of individual footnotes extracted from the text block.
        """
        lines = text.split('\n')
        page_footnotes = []

        footnote_pattern = self.pattern
        for i, line in enumerate(lines):
            match = re.match(footnote_pattern, line)
            if match:
                footnote_text = match.group(self.footnote_group)
                page_footnotes.append(footnote_text)
            elif i == 0:
                page_footnotes.append(line)
            else:
                page_footnotes[-1] += line

        return page_footnotes

if __name__ == '__main__':
    # Example usage:
    # Initialize the extractor with the chosen strategy
    footnote_extractor = FootnoteExtractor()

    # Open a document to work with
    doc_path = "C:/Users/sandra.eiche/Documents/kood/parser_comparison/varia_data/20181206-tp525-3-1-the-us-army-in-mdo-2028-final.pdf"
    doc = fitz.open(doc_path)

    # Extract tables from a specific page
    page = doc.load_page(25)  # Load the first page
    footnotes = footnote_extractor.extract_footnotes_from_page(page)
    print(footnotes)

    # Extract tables from the entire document
    footnotes_from_doc = footnote_extractor.extract_footnotes_from_doc(doc)
    print(footnotes_from_doc)
