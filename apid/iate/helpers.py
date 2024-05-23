import time
import json
import requests
import re
from . import authentication_requests
from . import iate_requests
import pandas as pd


def process_entry(entry, domains, target_languages):
    processed_entries = []
    domain_hierarchy = []
    for domain in entry['domains']:
        domain_hierarchy.append(" > ".join(get_domain_hierarchy_by_code(domains, domain['code'])))

    domain_hierarchy_str = "; ".join(domain_hierarchy)

    for tl in target_languages:

        if tl in entry['language']:
            lang_data = entry['language'][tl]
            term_entries = lang_data.get('term_entries', [])

            for term_entry in term_entries:
                creation_time = entry['metadata']['creation']['timestamp'].split('T')[0]
                modification_time = entry['metadata']['modification']['timestamp'].split('T')[0]

                term_refs = ''
                if 'term_references' in term_entry:
                    term_refs = term_entry['term_references']
                    term_references = ''
                    for tf in term_refs: 
                        term_references += tf['text']
                        term_references += '; '
                    term_refs = term_references.strip('; ')
                    #cleaned_refs = [clean_term_source(item) for item in term_entry['term_references']]
                    #term_refs = '; '.join(cleaned_refs).strip('; ')
                
                def_refs = ''
                if 'definition_references' in lang_data:
                    def_refs = lang_data['definition_references']
                    def_references = ''
                    for df in def_refs: 
                        def_references += df['text']
                        def_references += '; '
                    def_refs = def_references.strip('; ')

                note_texts = ''

                if 'note' in lang_data:
                    note_texts += lang_data['note']['value']

                note_refs = ''

                if 'note_references' in lang_data:
                    for ref in lang_data['note_references']:
                        note_refs += ref['text']
                
                term_note_text = ''

                if 'note' in term_entry:
                    term_note_text += term_entry['note']['value']

                term_note_references = ''

                if 'note_references' in term_entry:
                    for ref in term_entry['note_references']:
                        term_note_references += ref['text']

                context_texts = ''

                if 'contexts' in term_entry:
                    for c in term_entry['contexts']:
                        context_texts += c['context']

                context_refs = ''

                if 'contexts' in term_entry:
                    for c in term_entry['contexts']:
                        context_refs += c['reference']['text']

                
                if 'a href="/entry/' in lang_data.get('definition', ''):
                    definition_with_link = lang_data.get('definition', '').replace('"/entry/', '"https://iate.europa.eu/entry/')
                else:
                    definition_with_link = lang_data.get('definition', '')

                processed_entry = {
                    'IATE link': '<a href="https://iate.europa.eu/entry/result/' + str(entry['id']) + '">' + str(entry['id']) + '</a>',
                    #'ID': str(entry['id']),
                    'Lisatud': creation_time,
                    'Muudetud': modification_time,
                    'Valdkond': domain_hierarchy_str,
                    'Keel': tl.upper(),
                    'Termin': term_entry['term_value'],
                    'Termini allikaviide': term_refs,
                    'Termini märkus': term_note_text,
                    'Termini märkuse allikaviide': term_note_references,
                    'Definitsioon': definition_with_link,
                    'Definitsiooni allikaviited': def_refs,
                    'Mõiste märkus': note_texts,
                    'Mõiste märkuse allikaviide': note_refs,
                    'Kasutusnäide': context_texts,
                    'Kasutusnäite allikaviide': context_refs
                    }
                
                processed_entries.append(processed_entry)

    return processed_entries


def search_results_to_dataframe(query, source_languages, target_languages, num_pages, optional_parameters):
    search_results_to_dataframe_algus = time.time()

    with requests.Session() as session:
        tokens = authentication_requests.get_iate_tokens(session=session)
        access_token = tokens['tokens'][0]['access_token']
        results_list = []

        yhe_otsingu_algus = time.time()
        results = iate_requests.perform_single_search(access_token, query, source_languages, target_languages, num_pages, session=session, **optional_parameters)
        yhe_otsingu_lopp = time.time()

        print(f'perform_single_search võttis aega {yhe_otsingu_lopp - yhe_otsingu_algus:.2f} sekundit')
        with open('apid/iate/data/domains.json', 'r', encoding='utf-8') as file:
            domains = json.load(file)

        for r in results:
            if 'self' in r:
                single_entity_algus = time.time()
                entry = iate_requests.get_single_entity_by_href(access_token, r['self']['href'], session=session)
                single_entity_lopp = time.time()
                print(f'get_single_entity_by_href võttis aega {single_entity_lopp - single_entity_algus:.2f} sekundit')
                processed_entries = process_entry(entry, domains, target_languages)
                results_list.extend(processed_entries)

    search_results_to_dataframe_lopp = time.time()

    print(f'search_results_to_dataframe võttis aega {search_results_to_dataframe_lopp - search_results_to_dataframe_algus:.2f} sekundit')

    return pd.DataFrame(results_list)
    

def get_domain_name_by_code(data, domain_code):
    def search_domain(domains):
        for domain in domains:
            if domain['code'] == domain_code:
                return domain['name']
            
            if 'subdomains' in domain:
                name = search_domain(domain['subdomains'])
                if name:
                    return name
    
    return search_domain(data['items']) or domain_code


def get_domain_hierarchy_by_code(data, domain_code, hierarchy=None):
    if hierarchy is None:
        hierarchy = []
    
    for item in data['items']:
        if item['code'] == domain_code:
            return hierarchy + [item['name']]
        
        if 'subdomains' in item:
            subdomain_result = get_domain_hierarchy_by_code({'items': item['subdomains']}, domain_code, hierarchy + [item['name']])
            if subdomain_result:
                return subdomain_result
    
    return None