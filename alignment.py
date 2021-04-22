from transformers import AutoTokenizer  # Or BertTokenizer
from transformers import MBartTokenizer
from data.scripts.anntools import Collection
from pathlib import Path
import json
import os


def extract_keyphrases(keyphrases, text, tokens):
    tags = {}
    for keyphrase in sorted(keyphrases, key=lambda x: len(x.text)):
        ktext = keyphrase.text
        ktokens = [text[s[0]:s[1]] for s in keyphrase.spans]

        is_found, spans = False, []
        idxs, ponteiro, cmp_token, cmp_idxs = [], 0, [], []
        for i, token in enumerate(tokens):
            if token == ktokens[ponteiro]:
                spans.append(token)
                idxs.append(i)
                ponteiro += 1
                cmp_token, cmp_idxs = [], []
            elif token.replace('##', '') in ktokens[ponteiro]:
                cmp_token.append(token.replace('##', ''))
                cmp_idxs.append(i)
                for j in range(len(cmp_token)):
                    if ''.join(cmp_token[j:]) == ktokens[ponteiro]:
                        spans.append(''.join(cmp_token[j:]))
                        idxs.extend(cmp_idxs[j:])
                        ponteiro += 1
                        cmp_token, cmp_idxs = [], []
                        break
            else:
                idxs = []
                cmp_token, cmp_idxs = [], []

            if ponteiro == len(ktokens):
                is_found = True
                break

        tags[keyphrase.id] = {
            'text': ktext,
            'idxs': idxs,
            'tokens': [text[s[0]:s[1]] for s in keyphrase.spans],
            'attributes': [attr.__repr__() for attr in keyphrase.attributes],
            'spans': keyphrase.spans,
            'label': keyphrase.label,
            'id': keyphrase.id,
            'error': not is_found
        }
    return tags


def add_data(c, data, tokenizer, ref=None):
    for i, instance in enumerate(c.sentences):
        text = instance.text
        tokens = tokenizer.convert_ids_to_tokens(tokenizer(text)['input_ids'])

        keyphrases = extract_keyphrases(instance.keyphrases, text, tokens)

        relations = []
        for relation in instance.relations:
            relations.append({
                'arg1': relation.origin,
                'arg2': relation.destination,
                'label': relation.label
            })

        data_dict = {'text': text,
                     'tokens': tokens,
                     'keyphrases': keyphrases,
                     'relations': relations
                     }
        if ref is not None:
            data_dict['language'] = ref['language']
            data_dict['domain'] = ref['domain']
        data.append(data_dict)


def run():
    with open('config_alignment.json') as f:
        config_file = json.load(f)
        pretrained_model_path = config_file['pretrained_model_path']
        input_path = config_file['input_path']
        output_file_name = config_file['output_file_name']
        is_ref = config_file['is_ref']

    if 'mbart' in pretrained_model_path:
        tokenizer = MBartTokenizer.from_pretrained('facebook/mbart-large-cc25')
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, do_lower_case=False)

    data = []
    if is_ref:
        for fname in Path(input_path).rglob("*.txt"):
            domain = fname.name.split('.')[0]
            language = 'spanish'
            if domain == 'cord':
                language = 'english'
            ref = {'language': language, 'domain': domain}
            c = Collection()
            c.load(fname)
            add_data(c, data, tokenizer, ref)
    else:
        c = Collection()
        c.load(Path(input_path + 'output.txt'))
        add_data(c, data, tokenizer)

    # Create output files
    json.dump(data, open(input_path + output_file_name, 'w'), sort_keys=True, indent=4,
              separators=(',', ':'))

if __name__ == '__main__':
    run()