from collections import defaultdict
import fitz  # PyMuPDF
import json
from parser_utils import calculate_content_box


class TableExtractor:
    """
    A class to extract tables from PDF documents using PyMuPDF (fitz).

    Attributes:
        strategy (str): The table extraction strategy to use.
        join_tolerance (int): The tolerance for joining tables.
        min_row_count (int): The minimum number of rows a table must have to be considered valid.
        min_col_count (int): The minimum number of columns a table must have to be considered valid.

    Methods:
        extract_tables_from_page(page, header_height=0, footer_height=0): Extracts tables from a single PDF page.
        extract_tables_from_doc(doc, header_height=0, footer_height=0): Extracts tables from an entire PDF document.
        __is_table(tab: fitz.table.Table, min_row_count: int =  2, min_col_count: int =  2): Checks if a table is valid based on row and column counts.
    """

    def __init__(self, table_extraction_strategy: str, join_tolerance: int = 1, min_row_count: int = 2, min_col_count: int = 2) -> None:
        """
        Initialize the TableExtractor with the specified extraction strategy and tolerances.
        """
        self.strategy = table_extraction_strategy
        self.join_tolerance = join_tolerance
        self.min_row_count = min_row_count
        self.min_col_count = min_col_count

    def extract_tables_from_page(self, page: fitz.Page, header_height: int = 0, footer_height: int = 0) -> str:
        """
        Extract tables from a single PDF page using the specified strategy and join_tolerance.

        Parameters:
            page (fitz.Page): The page from which to extract tables.
            header_height (int): The height of the page header to exclude from extraction.
            footer_height (int): The height of the page footer to exclude from extraction.

        Returns:
            str: A JSON string representing the extracted tables, or None if no tables were found.
        """
        # Extract tables from a single page using the specified strategy and join_tolerance.
        # The clip parameter is optional and can be used to specify a portion of the page.
        clip = None
        tables = list()
        table_bboxes = list()

        if header_height is not None and footer_height is not None:
            clip = calculate_content_box(page, header_height, footer_height)

        tables_found = page.find_tables(
            strategy=self.strategy,
            join_tolerance=self.join_tolerance,
            clip=clip
        )

        for tab in tables_found:
            if self.__is_table(tab, self.min_row_count, self.min_col_count):
                table_bbox = tab.bbox
                table = tab.to_pandas().to_dict('split')
                tables.append(table)
                table_bboxes.append(table_bbox)

        # Prepare the data for JSON serialization, including the page number.
        tables_json = [{
            "page_number": page.number + 1,  # Page numbers are zero-based in PyMuPDF
            "texts": tables,
            "bboxes": table_bboxes,
        }]
        if not tables:
            return None
        return json.dumps(tables_json)

    def extract_tables_from_doc(self, doc: fitz.Document, header_height: int =  0, footer_height: int =  0) -> str:
        """
        Extract tables from an entire PDF document.
        
        Parameters:
            doc (fitz.Document): The PDF document from which to extract tables.
            header_height (int): The height of the page header to exclude from extraction.
            footer_height (int): The height of the page footer to exclude from extraction.
        
        Returns:
            str: A JSON string representing all extracted tables from the document.
        """
        all_tables = []
        for page in doc:
            # Use the previously defined method to extract tables from each page
            tables_json = self.extract_tables_from_page(
                page, header_height=header_height, footer_height=footer_height)
            if tables_json:
                all_tables.extend(json.loads(tables_json))
        return json.dumps(all_tables)

    def __is_table(self, tab: fitz.table.Table, min_row_count: int = 2, min_col_count: int = 2):
        """
        Check if a table is valid based on the minimum number of rows and columns.
        
        Parameters:
            tab (fitz.table.Table): The table to validate.
            min_row_count (int): The minimum number of rows a table must have.
            min_col_count (int): The minimum number of columns a table must have.
        
        Returns:
            bool: True if the table is valid, False otherwise.
        """
        if tab.row_count < min_row_count or tab.col_count < min_col_count:
            return False
        return True


if __name__ == '__main__':
    # Example usage:
    # Initialize the extractor with the chosen strategy
    table_extractor = TableExtractor(
        table_extraction_strategy="lines_strict", join_tolerance=1)

    # Open a document to work with
    doc_path = "C://Users//sandra.eiche//Documents//kood//parser_comparison//ajp_data//20210310-AJP_5_with_UK_elem_final_web.pdf"
    doc = fitz.open(doc_path)

    # Extract tables from a specific page
    page = doc.load_page(0)  # Load the first page
    tables_from_page = table_extractor.extract_tables_from_page(page)
    print(tables_from_page)

    # Extract tables from the entire document
    tables_from_doc = table_extractor.extract_tables_from_doc(doc)
    print(tables_from_doc)
