import os
import time
import json
import requests
from app.controllers.iate_api_controllers import perform_single_search, get_single_entity_by_href, get_iate_domains
from app.controllers.token_controller import TokenController
import pandas as pd
import logging

logger = logging.getLogger('app')
logger.setLevel(logging.INFO)


token_controller = TokenController()

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
                
                def_refs = ' — '
                
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

                note_refs = ' — '

                if 'note_references' in lang_data:
                    for ref in lang_data['note_references']:
                        note_refs += ref['text']
                
                term_note_text = ''

                if 'note' in term_entry:
                    term_note_text += term_entry['note']['value']

                term_note_references = ' — '

                if 'note_references' in term_entry:
                    for ref in term_entry['note_references']:
                        term_note_references += ref['text']

                context_texts = ''

                if 'contexts' in term_entry:
                    for c in term_entry['contexts']:
                        context_texts += c['context']

                context_refs = ' — '

                if 'contexts' in term_entry:
                    for c in term_entry['contexts']:
                        context_refs += c['reference']['text']

                
                if 'a href="/entry/' in lang_data.get('definition', ''):
                    definition_with_link = lang_data.get('definition', '').replace('"/entry/', '"https://iate.europa.eu/entry/')
                else:
                    definition_with_link = lang_data.get('definition', '')

                processed_entry = {
                    'Link': '<a href="https://iate.europa.eu/entry/result/' + str(entry['id']) + '">' + str(entry['id']) + '</a>',
                    'Lisatud': creation_time,
                    'Muudetud': modification_time,
                    'Valdkond': domain_hierarchy_str,
                    'Keel': tl.upper(),
                    'Termin': term_entry['term_value'],
                    'Termini allikas': term_refs,
                    'Definitsioon': definition_with_link + def_refs.strip(' — '),
                    'Termini märkus': term_note_text + term_note_references.strip(' — '),
                    'Märkus': note_texts + note_refs.strip(' — '),
                    'Kontekst': context_texts + context_refs.strip(' — '),
                }
                
                processed_entries.append(processed_entry)

    return processed_entries

def search_results_to_dataframe(query, source_languages, target_languages, only_first_batch, optional_parameters, domains):
    with requests.Session() as session:
        access_token = token_controller.get_access_token()
        results_list = []

        results = perform_single_search(access_token, query, source_languages, target_languages, only_first_batch, session=session, **optional_parameters)

        for r in results:
            if 'self' in r:
                
                entry = get_single_entity_by_href(access_token, r['self']['href'], session=session)

                processed_entries = process_entry(entry, domains, target_languages)
                results_list.extend(processed_entries)

    return pd.DataFrame(results_list)


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


def initialize_domains():
    logger.info("Fetching domains from IATE API...")
    session = requests.Session()
    access_token = token_controller.get_access_token()
    domains = get_iate_domains(access_token, session)

    if isinstance(domains, dict):
        logger.info("Domains fetched successfully.")
        return domains
    else:
        logger.warning(f"Error fetching domains from IATE API: {domains}")
        return None