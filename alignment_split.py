from transformers import AutoTokenizer  # Or BertTokenizer
from data.scripts.anntools import Collection
from pathlib import Path
from random import Random
import json
import os
import numpy as np

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

    # Get shuffled data
    index_list = np.arange(len(data))
    Random(42).shuffle(index_list)
    data = np.array(data)
    sentence_list = np.array(c.sentences)
    data = data[index_list]
    sentence_list = sentence_list[index_list]

    # Get train data
    size = int(len(data)*0.2)
    trainset, _set = data[size:], data[:size]
    train_collection, _set_collection = sentence_list[size:], sentence_list[:size]

    # Get dev and test data
    size = int(len(_set)*0.5)
    devset, testset = _set[size:], _set[:size]
    dev_collection, test_collection = _set_collection[size:], _set_collection[:size]

    if not os.path.exists('data/preprocessed'):
        os.mkdir('data/preprocessed')

    # Create output files
    json.dump(list(trainset), open('data/preprocessed/trainset.json', 'w'), sort_keys=True, indent=4, separators=(',', ':'))
    json.dump(list(devset), open('data/preprocessed/devset.json', 'w'), sort_keys=True, indent=4, separators=(',', ':'))
    json.dump(list(testset), open('data/preprocessed/testset.json', 'w'), sort_keys=True, indent=4, separators=(',', ':'))
    Collection(list(train_collection)).dump(Path('data/preprocessed/train.txt'))
    Collection(list(dev_collection)).dump(Path('data/preprocessed/dev.txt'))
    Collection(list(test_collection)).dump(Path('data/preprocessed/test.txt'))