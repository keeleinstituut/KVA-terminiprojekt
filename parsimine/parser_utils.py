from collections import defaultdict
import fitz
import os


def is_inside_bbox(block, bbox):
    """Checks if a block is inside a given bounding box."""
    return (bbox[0] <= block[0] <= bbox[2]) and \
           (bbox[1] <= block[1] <= bbox[3]) and \
           (bbox[0] <= block[2] <= bbox[2]) and \
           (bbox[1] <= block[3] <= bbox[3])


def print_n_pages(fdir, fname, start_page=0, end_page=None, display='blocks', metadata=True, header_height: int = 50, footer_height: int = 50):
    """
    Faili esialgseks läbivaatuseks.
    `display` võib olla "blocks" (elementide koordinaatidega) või "text" (puhas tekst)
    """

    document_path = os.path.join(fdir, fname)

    with fitz.open(document_path) as doc:
        if metadata:
            print(fname)
            print(doc.metadata)

        if end_page is None or end_page >= doc.page_count:
            end_page = doc.page_count - 1

        for page_no in range(start_page, end_page + 1):
            page = doc.load_page(page_no)
            print(
                f'Height and width of page: {page.rect.height}, {page.rect.width}')
            print('footer_bbox', (0, page.rect.height -
                  footer_height, page.rect.width, page.rect.height))
            print('header_bbox', (0,  0, page.rect.width, header_height))
            text = page.get_text(display, sort=True)
            if display == 'blocks':
                for block in text:
                    print(block)
            else:
                print(text)
            print('-'*25, f'PAGE {page_no} BREAK', '-'*25)


def is_table(tab: fitz.table.Table, min_row_count: int = 2, min_col_count: int = 2):
    """ Tabel ei saa olla ainult päisega või ühe tulbaga"""
    if tab.row_count < min_row_count or tab.col_count < min_col_count:
        return False
    return True


def get_all_tables(doc: fitz.Document, header_height=None, footer_height=None, table_extraction_strategy='lines_strict'):
    """Salvestab kõik tabeli tunnustele vastavad tabelid lehekülgede kaupa"""

    tables = defaultdict(list)

    for page in doc:
        clip = None
        if header_height is not None and footer_height is not None:
            clip = calculate_content_box(page, header_height, footer_height)

        tabs = page.find_tables(
            strategy=table_extraction_strategy, join_tolerance=1, clip=clip)

        for tab in tabs:
            if is_table(tab):
                df = tab.to_pandas()
                df.attrs['bbox'] = tab.bbox
                df.attrs['page'] = page.number
                df.attrs['full_text'] = tab.extract()
                tables[page.number].append(df)

    return tables


def calculate_header_and_footer_box(page_obj: fitz.Page, header_height: int = 50, footer_height: int = 50):
    """ Pole kasutusel """
    header_bbox = (0,  0, page_obj.rect.width, header_height)
    footer_bbox = (0, page_obj.rect.height - footer_height,
                   page_obj.rect.width, page_obj.rect.height)
    return [header_bbox, footer_bbox]


def calculate_content_box(page_obj: fitz.Page, header_height: int = 50, footer_height: int = 50):
    return (0, header_height, page_obj.rect.width, page_obj.rect.height - footer_height)
