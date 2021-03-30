from data.scripts.anntools import Collection, Sentence, Keyphrase, Relation
from pathlib import Path
import time
import torch
import utils


def check_valid_token(cur_token):
    return not (cur_token.startswith('##') or cur_token == '[CLS]' or cur_token == '[SEP]')


def get_token_at_position(tokens, index):
    if tokens[index].startswith('##'):
        return ''
    result = tokens[index]
    i = index + 1
    while i < len(tokens) and tokens[i].startswith('##'):
        result += tokens[i].replace('##', '')
        i += 1
    return result


def get_multiword_dict(multiword_list):
    multiword_dict = {}
    for index_pair in multiword_list:
        index_from, index_to = index_pair[0], index_pair[1]
        # inferencer 1
        if index_from < index_to:
            value = multiword_dict.get(index_from, [])
            value.append(index_to)
            multiword_dict[index_from] = value
    return multiword_dict


def check_if_list_wholly_contained(list_a, list_b):
    i, j, equal = 0, 0, 0
    while i < len(list_a) and j < len(list_b):
        if list_a[i] == list_b[j]:
            equal += 1
            i += 1
            j += 1
        elif list_a[i][0] < list_b[j][0]:
            i += 1
        else:
            j += 1
    return equal >= len(list_a)

def discard_entities(sentence):
    """7. Discard entities that are wholly contained within another entity"""
    sentence.sort()
    to_remove = []
    last_span_list = []
    for keyphrase in sentence.keyphrases:
        if check_if_list_wholly_contained(keyphrase.spans, last_span_list):
            to_remove.append(keyphrase)
        else:
            last_span_list = keyphrase.spans
    # print('To remove: {}'.format(to_remove))
    for key_remove in to_remove:
        sentence.keyphrases.remove(key_remove)

def get_collection(preprocessed_dataset, entity, multiword, sameas, relation, relation_type):
    c = Collection()
    global_entity_id = 0
    for row, entity_list, multiword_list in zip(preprocessed_dataset, entity, multiword):
        if isinstance(multiword_list, torch.Tensor):
            multiword_list = multiword_list.detach().cpu().numpy()
        sentence_text = row['text']
        sentence = Sentence(sentence_text)
        tokens = row['tokens']
        # print(tokens)
        # print(entity_list)
        multiword_dict = get_multiword_dict(multiword_list)
        # print(multiword_dict)
        last_pos = 0
        for index, entity_id in enumerate(entity_list):
            if utils.entity_id2w[entity_id] != 'O' and check_valid_token(tokens[index]):
                cur_token = get_token_at_position(tokens, index)
                start = last_pos + sentence_text[last_pos:].find(cur_token)
                end = start + len(cur_token)
                span_list = [(start, end)]
                # print(cur_token)
                if index in multiword_dict:
                    for idx in multiword_dict[index]:
                        mw_token = get_token_at_position(tokens, idx)
                        if len(mw_token) > 0:
                            start = last_pos + sentence_text[last_pos:].find(mw_token)
                            end = start + len(mw_token)
                            span_list.append((start, end))
                # print(sentence_text[last_pos:])

                keyphrase = Keyphrase(sentence, utils.entity_id2w[entity_id], global_entity_id,
                                      span_list)
                # print(keyphrase)
                global_entity_id += 1
                sentence.keyphrases.append(keyphrase)
            if tokens[index] != '[CLS]' and tokens[index] != '[SEP]':
                last_pos += len(tokens[index].replace('##', ''))
        discard_entities(sentence)

        c.sentences.append(sentence)
    return c