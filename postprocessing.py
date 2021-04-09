from data.scripts.anntools import Collection, Sentence, Keyphrase, Relation
from pathlib import Path
import time
import torch
import utils
import pandas as pd


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


def get_bidirectional_relation_dict(relation_list):
    relation_dict = {}
    for index_from, index_to, label in relation_list:
        # inferencer 1
        # discarding reflexive relations
        if index_from < index_to and label >= 1:
            value = relation_dict.get(index_from, [])
            value.append(index_to)
            relation_dict[index_from] = value
    return relation_dict


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
    if label <= 0:
        return False
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
        if label_idx > 0 and token_idx1 in token2entity and token_idx2 in token2entity:
            origin, destination = token2entity[token_idx1], token2entity[token_idx2]
            if origin != destination and sentence.find_keyphrase(id=origin) is not None and sentence.find_keyphrase(
                    id=destination) is not None:
                sentence.relations.append(Relation(sentence, origin, destination, relation_id2w[label_idx]))


def get_collection(preprocessed_dataset, entity, multiword, sameas, related, relation_type):
    c = Collection()
    global_entity_id = 0
    for row, entity_list, multiword_list, sameas_list, related_list, relation_type_list in zip(preprocessed_dataset, entity, multiword, sameas, related, relation_type):
        if isinstance(multiword_list, torch.Tensor):
            entity_list = entity_list.detach().cpu().numpy()
            multiword_list = multiword_list.detach().cpu().numpy()
            sameas_list = sameas_list.detach().cpu().numpy()
            related_list = related_list.detach().cpu().numpy()
            relation_type_list = relation_type_list.detach().cpu().numpy()
        sentence_text = row['text']
        sentence = Sentence(sentence_text)
        tokens = row['tokens']
        # print(tokens)
        # print(entity_list)
        # print(multiword_list)
        multiword_dict = get_bidirectional_relation_dict(multiword_list)
        relation_type_list = filter_relations_using_related(relation_type_list, related_list)
        # print(multiword_dict)
        last_pos = 0
        token_index_to_entity_id = {}
        for index, entity_id in enumerate(entity_list):
            entity_index_list = []
            if utils.entity_id2w[entity_id] != 'O' and check_valid_token(tokens[index]):
                cur_token = get_token_at_position(tokens, index)
                start = last_pos + sentence_text[last_pos:].find(cur_token)
                end = start + len(cur_token)
                span_list = [(start, end)]
                entity_index_list.append(index)
                # print(cur_token)
                if index in multiword_dict:
                    for idx in multiword_dict[index]:
                        mw_token = get_token_at_position(tokens, idx)
                        if len(mw_token) > 0:
                            start = last_pos + sentence_text[last_pos:].find(mw_token)
                            end = start + len(mw_token)
                            span_list.append((start, end))
                            entity_index_list.append(idx)
                # print(sentence_text[last_pos:])

                keyphrase = Keyphrase(sentence, utils.entity_id2w[entity_id], global_entity_id,
                                      span_list)

                for entity_index in entity_index_list:
                    token_index_to_entity_id[entity_index] = global_entity_id

                # print(keyphrase)
                global_entity_id += 1
                sentence.keyphrases.append(keyphrase)
            if tokens[index] != '[CLS]' and tokens[index] != '[SEP]':
                last_pos += len(tokens[index].replace('##', ''))
        discard_entities(sentence)

        add_relations(sentence, sameas_list, token_index_to_entity_id, {1: 'same-as'})
        add_relations(sentence, relation_type_list, token_index_to_entity_id, utils.relation_id2w)

        c.sentences.append(sentence)
    return c

# CREATE TABLE METHODS
def get_field(text, field):
    line_list = text.split('{}='.format(field))
    if len(line_list) <= 1:
        return ''
    return line_list[1].strip("'").split("'")[0]

# RELATION
def get_relation_from_line(line):
    from_text = get_field(line, 'from')
    to_text = get_field(line, 'to')
    label_text = get_field(line, 'label')
    return from_text, to_text, label_text

def fill_relation_dict(table_dict, line):
    from_text, to_text, label_text = get_relation_from_line(line)
    if from_text.strip() == '':
        return
    table_dict['from'].append(from_text)
    table_dict['to'].append(to_text)
    table_dict['label'].append(label_text)

def create_relation_output_table(file_name):
    spurious, missing = False, False
    spurious_dict = {'from': [], 'to': [], 'label':[]}
    missing_dict = {'from': [], 'to': [], 'label':[]}
    with open(file_name, 'r') as f:
        for line in f:
            if line.startswith('====') and 'SPURIOUS_B' in line:
                spurious = True
                missing = False
                continue
            if line.startswith('====') and 'MISSING_B' in line:
                missing = True
                spurious = False
                continue
            if line.startswith('------------------'):
                missing = False
                spurious = False
                continue

            if spurious:
                fill_relation_dict(spurious_dict, line)

            if missing:
                fill_relation_dict(missing_dict, line)
    writer = pd.ExcelWriter('verbose_relation.xlsx', engine='xlsxwriter')
    relation_columns = ['from', 'to', 'label']
    df_spurious = pd.DataFrame(spurious_dict, columns=relation_columns)
    df_missing = pd.DataFrame(missing_dict, columns=relation_columns)
    df_spurious.to_excel(writer, sheet_name='spurious')
    df_missing.to_excel(writer, sheet_name='missing')
    writer.save()

# ENTITY
def get_entity_from_line(line):
    text = get_field(line, 'text')
    label = get_field(line, 'label')
    return text, label

def get_two_entity_from_line(line):
    predicted, gold = line.split('-->')[0], line.split('-->')[-1]
    predicted_text = get_field(predicted, 'text')
    predicted_label = get_field(predicted, 'label')
    gold_text = get_field(gold, 'text')
    gold_label = get_field(gold, 'label')
    return predicted_text, predicted_label, gold_text, gold_label

def fill_entity_dict(table_dict, line):
    text, label = get_entity_from_line(line)
    if text.strip() == '':
        return
    table_dict['text'].append(text)
    table_dict['label'].append(label)

def fill_two_entity_dict(table_dict, line):
    predicted_text, predicted_label, gold_text, gold_label = get_two_entity_from_line(line)
    if predicted_text.strip() == '':
        return
    table_dict['predicted_text'].append(predicted_text)
    table_dict['predicted_label'].append(predicted_label)
    table_dict['gold_text'].append(gold_text)
    table_dict['gold_label'].append(gold_label)

def create_entity_output_table(file_name):
    spurious, missing, incorrect, partial = False, False, False, False
    spurious_dict = {'text': [], 'label': []}
    missing_dict = {'text': [], 'label': []}
    incorrect_dict = {'predicted_text': [], 'predicted_label': [], 'gold_text': [], 'gold_label': []}
    partial_dict = {'predicted_text': [], 'predicted_label': [], 'gold_text': [], 'gold_label': []}
    with open(file_name, 'r') as f:
        # order: incorrect -> partial -> spurious -> missing
        for line in f:
            if line.startswith('====') and 'INCORRECT_A' in line:
                incorrect = True
                continue
            if line.startswith('====') and 'PARTIAL_A' in line:
                partial = True
                incorrect = False
                continue
            if line.startswith('====') and 'SPURIOUS_A' in line:
                spurious = True
                partial = False
                continue
            if line.startswith('====') and 'MISSING_A' in line:
                missing = True
                spurious = False
                continue
            if line.startswith('------------------'):
                missing = False
                continue

            if spurious:
                fill_entity_dict(spurious_dict, line)
            elif missing:
                fill_entity_dict(missing_dict, line)
            elif incorrect:
                fill_two_entity_dict(incorrect_dict, line)
            elif partial:
                fill_two_entity_dict(partial_dict, line)

    writer = pd.ExcelWriter('verbose_entity.xlsx', engine='xlsxwriter')
    two_entity_columns = ['predicted_text', 'predicted_label', 'gold_text', 'gold_label']
    entity_columns = ['text', 'label']
    df_incorrect = pd.DataFrame(incorrect_dict, columns=two_entity_columns)
    df_partial = pd.DataFrame(partial_dict, columns=two_entity_columns)
    df_spurious = pd.DataFrame(spurious_dict, columns=entity_columns)
    df_missing = pd.DataFrame(missing_dict, columns=entity_columns)

    df_incorrect.to_excel(writer, sheet_name='incorrect')
    df_partial.to_excel(writer, sheet_name='partial')
    df_spurious.to_excel(writer, sheet_name='spurious')
    df_missing.to_excel(writer, sheet_name='missing')
    writer.save()

