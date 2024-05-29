from copy import deepcopy
import pandas as pd
from app.controllers.mw_api_controllers import get_data


def refine_results(original_results):
    refined_results = {}
    for index, result in enumerate(original_results, start=1):
        result_copy = deepcopy(result)
        if 'meta' in result_copy:
            del result_copy['meta']
        if 'hwi' in result_copy and 'prs' in result_copy['hwi']:
            del result_copy['hwi']['prs']
        if 'uros' in result_copy:
            for i in result_copy['uros']:
                if 'prs' in i:
                    del i['prs']
        if 'date' in result_copy:
            del result_copy['date']
        refined_results[f'result_{index}'] = result_copy
    return refined_results


def json_to_df_with_definitions_and_usages(query):
    result = get_data(query, "collegiate")
    refined_json = refine_results(result)

    rows = []

    def clean_text(text):
        import re
        text = text.replace('{bc}', '').replace('{wi}', '').replace('{/wi}', '').replace('{it}', '').replace('{/it}', '').replace('{parahw}', '').replace('{/parahw}', '').replace('*', '')
        text = re.sub(r'\{sx\|[^}]*\|\|\}', '', text)
        text = text.strip()
        return text

    def extract_definitions(def_list):
        definitions = []
        for definition in def_list:
            if 'sseq' in definition:
                for sseq_item in definition['sseq']:
                    for sense in sseq_item:
                        if 'dt' in sense[1]:
                            for dt_item in sense[1]['dt']:
                                if dt_item[0] == 'text':
                                    cleaned_def = clean_text(dt_item[1])
                                    definitions.append(cleaned_def)
        return '; '.join(definitions)

    def extract_verbal_illustrations(def_list):
        verbal_illustrations = []
        for definition in def_list:
            if 'sseq' in definition:
                for sseq_item in definition['sseq']:
                    for sense in sseq_item:
                        if 'dt' in sense[1]:
                            for dt_item in sense[1]['dt']:
                                if dt_item[0] == 'vis':
                                    for vis_item in dt_item[1]:
                                        cleaned_vis = clean_text(vis_item['t'])
                                        verbal_illustrations.append(cleaned_vis)
        return '; '.join(verbal_illustrations)

    for result_key, result_value in refined_json.items():
        if not isinstance(result_value, dict):
            continue

        headword = clean_text(result_value.get('hwi', {}).get('hw', ''))
        if not headword:
            return refined_json

        row = {
            'Allikas': 'Merriam-Webster',
            'Keelend': headword,
            'Definitsioon': extract_definitions(result_value.get('def', [])),
            'Lühike definitsioon': '; '.join(result_value.get('shortdef', [])),
            'Näide': extract_verbal_illustrations(result_value.get('def', []))
        }
        rows.append(row)

    return pd.DataFrame(rows)