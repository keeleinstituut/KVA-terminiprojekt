import xml.etree.ElementTree as ET
import pandas as pd
from . import entries_requests


def print_dictionary_titles(results):
    for r in results:
        print(r['dictionaryName'])


def print_search_results(results):
    if 'results' in results:
        for r in results['results']:
            print(r)


def print_suggestions(results):
    if 'suggestions' in results:
        for s in results['suggestions']:
            print(s)


def print_entry_data(results):
    if 'entryContent' in results:
        root = ET.fromstring(results['entryContent'])
        orth_value = root.find('.//orth').text
        print(orth_value)
        definitions = [def_elem.text for def_elem in root.findall('.//def')]

        for i, definition in enumerate(definitions, 1):
            print(f"{i}. {definition}")



def entry_data_to_dataframe(dict_code, search_word, page_size, page_index):
    search_results = entries_requests.get_search_results(dict_code, search_word, page_size, page_index)
    
    data = []

    if 'results' in search_results:
        for r in search_results['results']:
            url = r['entryUrl'] + '?format=xml'
            result_data = entries_requests.get_entry_by_entry_url(url)

            if 'entryContent' in result_data:
                root = ET.fromstring(result_data['entryContent'])
                orth_value = root.find('.//orth').text
                definitions = [def_elem.text for def_elem in root.findall('.//def')]

                for definition in definitions:
                    data.append({
                        'Allikas': 'Collins',
                        'Keelend': orth_value, 
                        'Definitsioon': definition,
                        'Lühike definitsioon': None,
                        'Näide': None,
                        'Kasutus': None
                    })

    df = pd.DataFrame(data)
    
    return df