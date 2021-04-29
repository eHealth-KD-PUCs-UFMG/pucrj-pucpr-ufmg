from data.scripts.anntools import Collection, Sentence, Keyphrase, Relation
import torch
import utils
from nltk.corpus import stopwords
import string
stop = stopwords.words('spanish')
stop += stopwords.words('english')

def check_if_stopword(token):
    return token in stop

def check_if_connector(token):
    return token in '-+_%'

def check_valid_token(cur_token):
    return not (cur_token.startswith('##') or cur_token == '[CLS]' or cur_token == '[SEP]'
                or check_if_connector(cur_token))

def check_valid_initial_token(cur_token):
    return check_valid_token(cur_token) and not check_if_stopword(cur_token) and not cur_token in string.punctuation


def get_token_at_position(tokens, index):
    if not check_valid_token(tokens[index]):
        return ''
    result = tokens[index]
    i = index + 1
    while i < len(tokens) and (tokens[i].startswith('##') or check_if_connector(tokens[i])):
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

def add_relations(sentence, relation_list, token2entity, relation_id2w):
    for token_idx1, token_idx2, label_idx in relation_list:
        relation_label = relation_id2w[label_idx]
        relation_idx1 = token_idx1
        relation_idx2 = token_idx2
        # if relation is INV, change label and invert indexes
        if relation_label.endswith('_INV'):
            relation_label = relation_label.split('_')[0]
            relation_idx1 = token_idx2
            relation_idx2 = token_idx1

        if relation_label != 'NONE' and relation_idx1 in token2entity and relation_idx2 in token2entity:
            origin, destination = token2entity[relation_idx1], token2entity[relation_idx2]
            if origin != destination and sentence.find_keyphrase(id=origin) is not None and sentence.find_keyphrase(
                    id=destination) is not None:
                sentence.relations.append(Relation(sentence, origin, destination, relation_label))

def check_if_contiguous_entity(index, entity_id, entity_list, tokens):
    return index + 1 < len(entity_list)\
           and ((utils.entity_id2w[entity_list[index + 1]].strip('-BI') == utils.entity_id2w[entity_id].strip('-BI')
                and utils.entity_id2w[entity_list[index + 1]].startswith('I-'))
                or not check_valid_token(tokens[index + 1]))

def fix_span_list(span_list):
    if len(span_list) == 0:
        return span_list
    result = []
    last_span = span_list[0]
    for i in range(1, len(span_list)):
        last_start, last_end = last_span
        cur_start, cur_end = span_list[i]
        if last_end == cur_start:
            last_span = (last_start, cur_end)
        else:
            result.append(last_span)
            last_span = span_list[i]
    result.append(last_span)
    return result

def find_nth_occurrence(text, token, num_occurrence):
    val = -1
    for i in range(num_occurrence):
        val = text.find(token, val + 1)
    return val


def get_collection(preprocessed_dataset, entity, related, relations_inv=False):
    c = Collection()
    global_entity_id = 0
    for row, entity_list, related_list in zip(preprocessed_dataset, entity, related):
        if isinstance(entity_list, torch.Tensor):
            entity_list = entity_list.detach().cpu().numpy()
            related_list = related_list.detach().cpu().numpy()
        sentence_text = row['text']
        sentence = Sentence(sentence_text)
        tokens = row['tokens']
        token_index_to_entity_id = {}
        count_token = {}
        index = 0
        while index < len(entity_list):
            # print(tokens[index])
            entity_id = entity_list[index]
            # print(entity_id)
            entity_index_list = []
            if utils.entity_id2w[entity_id].startswith('B-') and check_valid_initial_token(tokens[index]):
                cur_token = get_token_at_position(tokens, index)
                # print('found token: %s' % cur_token)
                # start = last_pos + sentence_text[last_pos:].find(cur_token)
                count = count_token.get(cur_token, 0)
                count_token[cur_token] = count + 1
                start = find_nth_occurrence(sentence_text, cur_token, count_token[cur_token])
                end = start + len(cur_token)
                span_list = [(start, end)]
                entity_index_list.append(index)

                while check_if_contiguous_entity(index, entity_id, entity_list, tokens):
                    index += 1
                    mw_token = get_token_at_position(tokens, index)
                    if len(mw_token) > 0:
                        # print('contiguous entities: %s' % mw_token)
                        # start = last_pos + sentence_text[last_pos:].find(mw_token)
                        count = count_token.get(mw_token, 0)
                        count_token[mw_token] = count + 1
                        start = find_nth_occurrence(sentence_text, cur_token, count_token[mw_token])
                        end = start + len(mw_token)
                        span_list.append((start, end))
                        entity_index_list.append(index)


                keyphrase = Keyphrase(sentence, utils.entity_id2w[entity_id].strip('-BI'), global_entity_id,
                                      fix_span_list(span_list))

                # print(keyphrase)
                for entity_index in entity_index_list:
                    token_index_to_entity_id[entity_index] = global_entity_id

                # print(keyphrase)
                global_entity_id += 1
                sentence.keyphrases.append(keyphrase)
            index += 1
        discard_entities(sentence)

        relation_id2w_local = utils.relation_id2w
        if relations_inv:
            relation_id2w_local = utils.relation_inv_id2w
        add_relations(sentence, related_list, token_index_to_entity_id, relation_id2w_local)

        c.sentences.append(sentence)
    return c