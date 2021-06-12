import pandas as pd
import os
import time
from shutil import copyfile

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

def create_relation_output_table(input_file_name, output_file_name):
    spurious, missing = False, False
    spurious_dict = {'from': [], 'to': [], 'label':[]}
    missing_dict = {'from': [], 'to': [], 'label':[]}
    with open(input_file_name, 'r') as f:
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
    writer = pd.ExcelWriter(output_file_name, engine='xlsxwriter')
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

def create_entity_output_table(input_file_name, output_file_name):
    spurious, missing, incorrect, partial = False, False, False, False
    spurious_dict = {'text': [], 'label': []}
    missing_dict = {'text': [], 'label': []}
    incorrect_dict = {'predicted_text': [], 'predicted_label': [], 'gold_text': [], 'gold_label': []}
    partial_dict = {'predicted_text': [], 'predicted_label': [], 'gold_text': [], 'gold_label': []}
    with open(input_file_name, 'r') as f:
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

    writer = pd.ExcelWriter(output_file_name, engine='xlsxwriter')
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

def generate_output_folder(trainer):
    # create folder
    time_str = time.strftime("%Y_%m_%d-%H:%M:%S")
    output_folder = 'output/{}/'.format(time_str)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # copy data file
    data_file = 'data/original/eval/{}/scenario1-main/output.txt'.format(trainer.eval_mode)
    copyfile(data_file, output_folder + 'data.txt')

    # output model parameters
    model_parameter_file_name = output_folder + 'model_parameters.txt'
    with open(model_parameter_file_name, 'w') as f:
        f.write(str(trainer))

    # entity and relation result raw files
    entity_file_name = output_folder + 'raw_verbose_entity.txt'
    relation_file_name = output_folder + 'raw_verbose_relation.txt'
    all_file_name = output_folder + 'evaluation.txt'
    trainer.eval(result_file_name=entity_file_name, verbose=True, scenario=2)
    trainer.eval(result_file_name=relation_file_name, verbose=True, scenario=3)
    trainer.eval(result_file_name=all_file_name)

    # entity and relation tables
    entity_table_name = output_folder + 'verbose_entity.xlsx'
    relation_table_name = output_folder + 'verbose_relation.xlsx'
    create_entity_output_table(entity_file_name, entity_table_name)
    create_relation_output_table(relation_file_name, relation_table_name)