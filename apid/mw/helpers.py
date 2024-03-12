from . import entries_requests
import json

def search_results_to_json(result):
    print(len(result))
    results_json = {}

    for r in result:
        hom_nr = r.get('hom', 1) 
        if hom_nr not in results_json:

            results_json[hom_nr] = {
                'headword': '',
                'variants': [],
                'functional_labels': '',
                'general_labels': [],
                'definitions': []
            }

        # Headword
        if 'hwi' in r:
            results_json[hom_nr]['headword'] = r['hwi']['hw']

        # Variants
        if 'vrs' in r:
            variants = []
            for v in r['vrs']:
                variant_pair = {v['vl']: v['va']}
                variants.append(variant_pair)
            results_json[hom_nr]['variants'] = variants

        # Functional labels
        if 'fl' in r:
            results_json[hom_nr]['functional_labels'] = r['fl']

        # General labels
        if 'lbs' in r:
            general_labels = [v for v in r['lbs']]
            results_json[hom_nr]['general_labels'] = general_labels

        # Definition section
        if 'def' in r:
            definitions = []
            for d in r['def']:
                def_item = {'vd': '', 'senses': []}
                if 'vd' in d:
                    def_item['vd'] = d['vd']
                if 'sseq' in d:
                    for sseq in d['sseq']:
                        for sense in sseq:
                            sense_data = {}
                            if 'sn' in sense[1]:
                                sense_data['sense_nr'] = sense[1]['sn']
                            if 'dt' in sense[1]:
                                dt_list = []
                                for dt in sense[1]['dt']:
                                    if dt[0] == 'text':
                                        dt_list.append({'text': dt[1]})
                                    elif dt[0] == 'vis':
                                        vis_list = [{'t': v['t']} for v in dt[1]]
                                        dt_list.append({'verbal_illustrations': vis_list})
                                sense_data['defining_text'] = dt_list
                            def_item['senses'].append(sense_data)
                definitions.append(def_item)
            results_json[hom_nr]['definitions'] = definitions

    return results_json
