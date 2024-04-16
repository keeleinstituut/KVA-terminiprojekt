import fitz  # Import PyMuPDF
import json
import re  # Import the regular expression module
from parser_utils import calculate_content_box

class RegexBlockExtractor:

    def __init__(self, regex_pattern: re.Pattern = re.compile(r"(\n\s*\n|^)((.|\n)*?)(?=\n\s*\n|$)", flags=re.X), regex_group = 1, consider_block_borders = False):
        self.regex_pattern = regex_pattern
        self.regex_group = regex_group
        self.consider_block_borders = consider_block_borders

    def extract_text_by_page(self, doc, page_ranges, header_height = 0, footer_height = 0):
        """
        Extracts full page texts from specified page ranges in a PDF document and segments the text using the class's regex pattern.

        Args:
        - doc (fitz.Document): A PyMuPDF Document object.
        - page_ranges (list of tuples): Each tuple contains two integers, start and end, defining a range of pages.

        Returns:
        - str: JSON string containing page numbers and segmented lists of texts from those pages.
        """
        results = []
        pattern = self.regex_pattern

        for start, end in page_ranges:
            for page_num in range(start, end + 1):  # +1 to include the end page
                if page_num < doc.page_count:  # Check if the page number is within the document
                    page = doc.load_page(page_num)
                    if not self.consider_block_borders:
                        full_text = page.get_text(clip=calculate_content_box(page, header_height=header_height, footer_height=footer_height))  # Extract full page text
                        segments = pattern.findall(full_text)  # Segment text based on the class's regex pattern
                        results.append({"page_number": page_num + 1, "texts": [segment[self.regex_group] for segment in segments]})  # +1 to make page number human-readable
                    else:
                        blocks = page.get_text("blocks", clip=calculate_content_box(page, header_height=header_height, footer_height=footer_height))
                        block_texts = [block[4] for block in blocks if block[4].strip()]
                        texts = []
                        for block_text in block_texts:
                            print(block_text)
                            segments = pattern.findall(block_text)
                            if not segments:
                                texts.append(block_text)
                                continue
                            texts.extend([segment[self.regex_group] for segment in segments])
                        results.append({"page_number": page_num + 1, "texts": texts})  # +1 to make page number human-readable

        # Convert the results to JSON format
        json_result = json.dumps(results, indent=2)
        return json_result


class BlockExtractor:
    @staticmethod
    def extract_text_by_page(doc, page_ranges, header_height = 0, footer_height = 0):
        """
        Extracts texts by block from specified page ranges in a PDF document.

        Args:
        - doc (fitz.Document): A PyMuPDF Document object.
        - page_ranges (list of tuples): Each tuple contains two integers, start and end, defining a range of pages.

        Returns:
        - str: JSON string containing page numbers and lists of texts from those pages.
        """
        results = []

        for start, end in page_ranges:
            for page_num in range(start, end + 1):  # +1 to include the end page
                if page_num < doc.page_count:  # Check if the page number is within the document
                    page = doc.load_page(page_num)
                    blocks = page.get_text("blocks", clip=calculate_content_box(page, header_height=header_height, footer_height=footer_height))  # Extract text by blocks
                    texts = [block[4] for block in blocks if block[4].strip()]  # Extract non-empty texts
                    results.append({"page_number": page_num + 1, "texts": texts})  # +1 to make page number human-readable

        # Convert the results to JSON format
        json_result = json.dumps(results, indent=2)
        return json_result


if __name__ == "__main__":
    # Example usage of BlockExtractor
    doc_path = "C:/Users/sandra.eiche/Documents/kood/parser_comparison/varia_data/20181206-tp525-3-1-the-us-army-in-mdo-2028-final.pdf"
    page_ranges = []  # Define your page ranges here

    # Open the document outside of the BlockExtractor
    doc = fitz.open(doc_path)

    # Call the static method and print the result
    json_output = BlockExtractor.extract_text_by_page(doc, page_ranges, header_height=50, footer_height=50)
    print(json_output)

    # Close the document when done
    doc.close()



    # Example usage of RegexBlockExtractor
    doc_path = "C://Users//sandra.eiche//Documents//kood//parser_comparison//varia_data//aap6.pdf"
    page_ranges = [(50, 50)]  # Define your page ranges here

    # Optionally, modify the class variable if needed
    # PageTextExtractor.regex_pattern = r"YourNewPatternHere"

    # Open the document
    doc = fitz.open(doc_path)

    # Call the class method and print the result
    regex_extractor = RegexBlockExtractor()
    json_output = regex_extractor.extract_text_by_page(doc, page_ranges, header_height=50, footer_height=50)
    print(json_output)

    # Close the document when done
    doc.close()