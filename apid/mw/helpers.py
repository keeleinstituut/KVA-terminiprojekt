from copy import deepcopy
import pandas as pd
import re


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


def json_to_df_with_definitions_and_usages(json_data):
    rows = []

    def clean_text(text):
        import re  # Make sure to import the re module for regex operations
        text = text.replace('{bc}', '').replace('{wi}', '').replace('{/wi}', '').replace('{it}', '').replace('{/it}', '').replace('{parahw}', '').replace('{/parahw}', '')
        text = re.sub(r'\{sx\|[^}]*\|\|\}', '', text)
        text = text.strip()
        return text

    def extract_cxs(cxs_list):
        cxs_text = []
        for cxs_item in cxs_list:
            cxs_label = cxs_item.get('cxl', '')
            cxtis_text = ', '.join([clean_text(cxti.get('cxt', '')) for cxti in cxs_item.get('cxtis', [])])
            cxs_text.append(f"{cxs_label} {cxtis_text}")
        return '; '.join(cxs_text)

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

    def extract_usages(usages_list):
        usages = []
        for usage in usages_list:
            pl_text = clean_text(usage.get('pl', ''))
            pt_texts = [clean_text(pt_item[1]) for pt_item in usage.get('pt', []) if pt_item[0] == 'text']
            usages.append(f"{pl_text} {' '.join(pt_texts)}")
        return '; '.join(usages)
    
     

    for result_key, result_value in json_data.items():
        if not isinstance(result_value, dict):
            continue

        headword = clean_text(result_value.get('hwi', {}).get('hw', ''))
        if not headword:
            return json_data

        row = {
            'Source': 'Merriam-Webster',
            'Headword': headword,
            'Homonym': result_value.get('hom', ''),
            'Functional Label': result_value.get('fl', ''),
            'Cross References': extract_cxs(result_value.get('cxs', [])),
            'Short Definitions': '; '.join(result_value.get('shortdef', [])),
            'Long Definitions': extract_definitions(result_value.get('def', [])),
            'Verbal Illustrations': extract_verbal_illustrations(result_value.get('def', [])),
            'Usages': extract_usages(result_value.get('usages', []))
        }
        rows.append(row)

    return pd.DataFrame(rows)