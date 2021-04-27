from data.scripts.anntools import Collection, Sentence, Keyphrase, Relation
from pathlib import Path
import time
import torch
from utils import relation_id2w, relation_w2id, entity_id2w, entity_w2id


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

def check_tuple_in_list(relation_tuple, related_list):
    idx1, idx2, label = relation_tuple
    for related_idx1, related_idx2, related_label in related_list:
        if related_label >= 1 and idx1 == related_idx1 and idx2 == related_idx2:
            return True
    return False

def filter_relations_using_related(relation_type_list, related_list):
    result = []
    for relation_tuple in relation_type_list:
        if check_tuple_in_list(relation_tuple, related_list):
            result.append(relation_tuple)
    return result


def add_relations(sentence, relation_list, token2entity, relation_id2w):
    for token_idx1, token_idx2, label_idx in relation_list:
        if relation_id2w[label_idx] != 'NONE' and token_idx1 in token2entity and token_idx2 in token2entity:
            origin, destination = token2entity[token_idx1], token2entity[token_idx2]
            if origin != destination and sentence.find_keyphrase(id=origin) is not None and sentence.find_keyphrase(
                    id=destination) is not None:
                sentence.relations.append(Relation(sentence, origin, destination, relation_id2w[label_idx]))

def check_if_contiguous_entity(index, entity_id, entity_list, tokens):
    return index + 1 < len(entity_list)\
           and ((entity_id2w[entity_list[index + 1]].strip('-BI') == entity_id2w[entity_id].strip('-BI')
                and entity_id2w[entity_list[index + 1]].startswith('I-'))
                or not check_valid_token(tokens[index + 1]))

def get_collection(preprocessed_dataset, entity, related):
    c = Collection()
    global_entity_id = 0
    for row, entity_list, related_list in zip(preprocessed_dataset, entity, related):
        if isinstance(entity_list, torch.Tensor):
            entity_list = entity_list.detach().cpu().numpy()
            related_list = related_list.detach().cpu().numpy()
        sentence_text = row['text']
        sentence = Sentence(sentence_text)
        tokens = row['tokens']
        # print(tokens)
        # print(entity_list)
        # print(multiword_list)
        # relation_type_list = filter_relations_using_related(relation_type_list, related_list)
        # print(multiword_dict)
        last_pos = 0
        token_index_to_entity_id = {}
        index = 0
        while index < len(entity_list):
            # print(tokens[index])
            entity_id = entity_list[index]
            # print(entity_id)
            entity_index_list = []
            if entity_id2w[entity_id] != 'O' and check_valid_token(tokens[index]):
                cur_token = get_token_at_position(tokens, index)
                # print('found token: %s' % cur_token)
                start = last_pos + sentence_text[last_pos:].find(cur_token)
                end = start + len(cur_token)
                span_list = [(start, end)]
                entity_index_list.append(index)
                last_pos += len(cur_token)

                while check_if_contiguous_entity(index, entity_id, entity_list, tokens):
                    index += 1
                    mw_token = get_token_at_position(tokens, index)
                    if len(mw_token) > 0:
                        # print('contiguous entities: %s' % mw_token)
                        start = last_pos + sentence_text[last_pos:].find(mw_token)
                        end = start + len(mw_token)
                        span_list.append((start, end))
                        entity_index_list.append(index)
                        last_pos += len(mw_token)

                keyphrase = Keyphrase(sentence, entity_id2w[entity_id].strip('-BI'), global_entity_id,
                                      span_list)

                # print(keyphrase)
                for entity_index in entity_index_list:
                    token_index_to_entity_id[entity_index] = global_entity_id

                # print(keyphrase)
                global_entity_id += 1
                sentence.keyphrases.append(keyphrase)
            elif tokens[index] != '[CLS]' and tokens[index] != '[SEP]':
                last_pos += len(tokens[index].replace('##', ''))
            index += 1
        discard_entities(sentence)

        add_relations(sentence, related_list, token_index_to_entity_id, relation_id2w)

        c.sentences.append(sentence)
    return c