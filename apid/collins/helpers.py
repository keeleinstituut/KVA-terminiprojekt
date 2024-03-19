import xml.etree.ElementTree as ET


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
