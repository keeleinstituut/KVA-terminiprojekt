import requests
import re
from . import authentication_requests
from . import catalogue_requests
from . import entries_requests
import pandas as pd


def print_languages():
    tokens = authentication_requests.get_iate_tokens()

    access_token = tokens['tokens'][0]['access_token']

    result = catalogue_requests.get_languages(access_token)

    if result:
        for item in result['items']:
            href = item['meta']['href']
            lang = catalogue_requests.get_single_entity_by_href(access_token, href)
            print(lang['name'], lang['code'], lang['is_official'])
    else:
        print('Tulemusi pole.')


def print_query_operators():
    tokens = authentication_requests.get_iate_tokens()

    access_token = tokens['tokens'][0]['access_token']

    result = catalogue_requests.get_query_operators(access_token)

    if result:
        if 'items' in result:
            for item in result['items']:
                href = item['meta']['href']
                lang = catalogue_requests.get_single_entity_by_href(access_token, href)
                print(lang['name'], lang['code'])
        else:
            print('Tulemusi pole.')
    else:
        print('Tulemusi pole.')


def print_searchable_fields():
    tokens = authentication_requests.get_iate_tokens()

    access_token = tokens['tokens'][0]['access_token']

    result = catalogue_requests.get_searchable_fields(access_token)

    if result:
        if 'items' in result:
            for item in result['items']:
                href = item['meta']['href']
                lang = catalogue_requests.get_single_entity_by_href(access_token, href)
                print(lang['name'], lang['code'])
        else:
            print('Tulemusi pole.')
    else:
        print('Tulemusi pole.')


def print_domains():
    tokens = authentication_requests.get_iate_tokens()

    access_token = tokens['tokens'][0]['access_token']

    result = catalogue_requests.get_domains(access_token)

    if result:
        for item in result['items']:
            print(item['code'], item['name'])
    else:
        print('Tulemusi pole.')


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


def print_term_types():
    tokens = authentication_requests.get_iate_tokens()

    access_token = tokens['tokens'][0]['access_token']

    result = catalogue_requests.get_term_types(access_token)

    if result:
        for item in result['items']:
            href = item['meta']['href']
            term_type = catalogue_requests.get_single_entity_by_href(access_token, href)
            print(term_type)
    else:
        print('Tulemusi pole.')


def print_reliabilities():
    tokens = authentication_requests.get_iate_tokens()

    access_token = tokens['tokens'][0]['access_token']

    result = catalogue_requests.get_reliabilities(access_token)

    if result:
        for item in result['items']:
            href = item['meta']['href']
            reliability = catalogue_requests.get_single_entity_by_href(access_token, href)
            print(reliability)
    else:
        print('Tulemusi pole.')


def create_reliabilites_list():

    reliabilites_list = []

    tokens = authentication_requests.get_iate_tokens()

    access_token = tokens['tokens'][0]['access_token']

    result = catalogue_requests.get_reliabilities(access_token)

    if result:
        for item in result['items']:
            href = item['meta']['href']
            reliability = catalogue_requests.get_single_entity_by_href(access_token, href)
            reliabilites_list.append(reliability)
    else:
        print('Tulemusi pole.')

    return reliabilites_list


def format_usage_examples(usages):
    value_for_df = ''
    cleaned_ref = ''

    for u in usages:
        cleaned_ref = re.sub(r'<a href="[^"]*" target="_blank">|</a>', '', u['reference']['text'])
        cleaned_ref = re.sub(r'<time datetime.*\">', '', cleaned_ref)
        cleaned_ref = cleaned_ref.replace('</time>', '')

        value_for_df += u['context'] + ' (' + cleaned_ref + '); '
    
    value_for_df = value_for_df.strip('; ').replace('<b>', '').replace('</b>', '').replace('&gt;', '>').replace('<br>', '').replace('<div>', '').replace('</div>', '')

    return value_for_df, cleaned_ref


def format_definition_or_note(definition_or_note):
    cleaned_def_or_note = re.sub(r'<a href="[^"]*" target="_blank">|</a>', '', definition_or_note)
    cleaned_def_or_note = re.sub(r'<time datetime="[^"]*">|</time>', '', cleaned_def_or_note)
    cleaned_def_or_note = cleaned_def_or_note.replace('<br>', '').replace('<p>', '').replace('</p>', '').replace('<i>', '').replace('</i>', '')

    if '"ENTRY_TO_ENTRY_CONVERTER"' in cleaned_def_or_note:
        cleaned_def_or_note = re.sub(r'\[ <a href="([^"]+)"[^>]*>IATE:(\d+) \]', r'[https://iate.europa.eu/entry/result/\2/all]', cleaned_def_or_note)

    return cleaned_def_or_note


def clean_term_source(term_source):
    cleaned_ref = term_source['text'].replace('<a href="', '').replace('" target="_blank">', ' - ')
    cleaned_ref = re.sub(r'<.*>', '', cleaned_ref)
    return cleaned_ref


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

                value_for_df, cleaned_ref = format_usage_examples(term_entry.get('contexts', []))

                term_refs = ''
                if 'term_references' in term_entry:
                    cleaned_refs = [clean_term_source(item) for item in term_entry['term_references']]
                    term_refs = '; '.join(cleaned_refs).strip('; ')
                
                def_refs = ''
                if 'definition_references' in lang_data:
                    cleaned_refs = [clean_term_source(item) for item in lang_data['definition_references']]
                    def_refs = '; '.join(cleaned_refs).strip('; ')

                processed_entry = {
                    'IATE link': 'https://iate.europa.eu/entry/result/' + str(entry['id']),
                    'Lisatud': creation_time,
                    'Muudetud': modification_time,
                    'Valdkond': domain_hierarchy_str,
                    'Keel': tl.upper(),
                    'Termin': term_entry['term_value'],
                    'Termini allikaviide': term_refs,
                    'Definitsioon': format_definition_or_note(lang_data.get('definition', '')),
                    'Definitsiooni allikaviited': def_refs,
                    'Märkus': format_definition_or_note(lang_data['note']['value']) if 'note' in lang_data else '',
                    'Kasutusnäide': value_for_df,
                    'Kasutusnäite allikaviide': cleaned_ref
                    }
                
                processed_entries.append(processed_entry)

    return processed_entries


def search_results_to_dataframe(query, source_language, target_languages, optional_parameters):
    with requests.Session() as session:
        tokens = authentication_requests.get_iate_tokens(session=session)
        access_token = tokens['tokens'][0]['access_token']
        results_list = []

        results = entries_requests.perform_single_search(access_token, query, source_language, target_languages, session=session, **optional_parameters)
        domains = catalogue_requests.get_domains(access_token, session=session)

        for r in results:
            if 'self' in r:
                entry = catalogue_requests.get_single_entity_by_href(access_token, r['self']['href'], session=session)
                processed_entries = process_entry(entry, domains, target_languages)
                results_list.extend(processed_entries)  # Extend list with all processed entries

        return pd.DataFrame(results_list)