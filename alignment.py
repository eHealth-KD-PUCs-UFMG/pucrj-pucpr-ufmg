from transformers import AutoTokenizer  # Or BertTokenizer
from data.scripts.anntools import Collection
from pathlib import Path
from random import shuffle
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

def run():
    path = 'dccuchile/bert-base-spanish-wwm-cased'
    tokenizer = AutoTokenizer.from_pretrained(path, do_lower_case=False)

    c = Collection()

    for fname in Path("data/original/training/").rglob("*.txt"):
        c.load(fname)

    data = []
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

        data.append({
            'text': text,
            'tokens': tokens,
            'keyphrases': keyphrases,
            'relations': relations
        })

    shuffle(data)
    size = int(len(data)*0.2)
    trainset, _set = data[size:], data[:size]

    size = int(len(_set)*0.5)
    devset, testset = _set[size:], _set[:size]

    if not os.path.exists('data/preprocessed'):
        os.mkdir('data/preprocessed')

    json.dump(trainset, open('data/preprocessed/trainset.json', 'w'), sort_keys=True, indent=4, separators=(',', ':'))
    json.dump(devset, open('data/preprocessed/devset.json', 'w'), sort_keys=True, indent=4, separators=(',', ':'))
    json.dump(testset, open('data/preprocessed/testset.json', 'w'), sort_keys=True, indent=4, separators=(',', ':'))