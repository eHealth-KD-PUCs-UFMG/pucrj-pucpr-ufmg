
import json
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import classification_report
import os
from shutil import copyfile
from pathlib import Path

from utils import relation_w2id, entity_w2id, ENTITIES, RELATIONS
from postprocessing import get_collection

class ProcDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (string): data
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Train:
    def __init__(self, model, criterion, optimizer, traindata, devdata, epochs, batch_size, batch_status=16,
                 early_stop=3, device='cuda', write_path='model.pt', eval_mode='develop',
                 pretrained_model='multilingual', log_path='logs', relations_positive_negative=False):
        self.epochs = epochs
        self.batch_size = batch_size
        self.batch_status = batch_status
        self.early_stop = early_stop
        self.device = device

        self.traindata = traindata
        self.devdata = devdata

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.write_path = write_path
        self.log_path = log_path
        if not os.path.exists(log_path):
            os.mkdir(log_path)

        self.eval_mode = eval_mode
        self.pretrained_model = pretrained_model

        # only use relations_positive_negative for train data
        self.traindata = DataLoader(self.preprocess(traindata, relations_positive_negative=relations_positive_negative),
                                    batch_size=batch_size, shuffle=True)
        self.devdata = DataLoader(self.preprocess(devdata), batch_size=batch_size, shuffle=True)

    def __str__(self):
        return "Epochs: {}\nBatch size: {}\nEarly stop: {}\nData: {}\nPretrained model: {}".format(self.epochs,
                                                                                                   self.batch_size,
                                                                                                   self.early_stop,
                                                                                                   self.eval_mode,
                                                                                                   self.pretrained_model)

    @staticmethod
    def _get_relations_data(row):
        sameas, relation, relation_type = [], [], []
        nrelations = 0
        for relation_ in row['relations']:
            try:
                arg1 = relation_['arg1']
                arg1_idx0 = row['keyphrases'][str(arg1)]['idxs'][0]

                arg2 = relation_['arg2']
                arg2_idx0 = row['keyphrases'][str(arg2)]['idxs'][0]

                label = relation_['label']
                if label == 'same-as':
                    sameas.append((arg1_idx0, arg2_idx0, 1))
                    sameas.append((arg2_idx0, arg1_idx0, 1))
                else:
                    relation.append((arg1_idx0, arg2_idx0, 1))
                    relation_type.append((arg1_idx0, arg2_idx0, relation_w2id[label]))
                    # negative same-as relation
                    sameas.append((arg1_idx0, arg2_idx0, 0))
                    sameas.append((arg2_idx0, arg1_idx0, 0))
            except:
                pass

        # negative relation examples
        arg1_idx0s = [w[0] for w in relation]
        arg2_idx0s = [w[1] for w in relation]
        for arg1_idx0 in arg1_idx0s:
            for arg2_idx0 in arg2_idx0s:
                f = [w for w in relation if w[0] == arg1_idx0 and w[1] == arg2_idx0]
                if len(f) == 0:
                    relation.append((arg1_idx0, arg2_idx0, 0))
                    nrelations += 1

                f = [w for w in relation if w[0] == arg2_idx0 and w[1] == arg1_idx0]
                if len(f) == 0:
                    relation.append((arg2_idx0, arg1_idx0, 0))
                    nrelations += 1
        return sameas, relation, relation_type

    @staticmethod
    def _get_relations_positive_negative_data(row):
        sameas, relation, relation_type = [], [], []
        for relation_ in row['relations_positive_negative']:
            try:
                arg1 = relation_['arg1']
                arg1_idx0 = row['keyphrases'][str(arg1)]['idxs'][0]

                arg2 = relation_['arg2']
                arg2_idx0 = row['keyphrases'][str(arg2)]['idxs'][0]

                label = relation_['label']
                if label == 'same-as':
                    sameas.append((arg1_idx0, arg2_idx0, 1))
                    sameas.append((arg2_idx0, arg1_idx0, 1))
                else:
                    if label == 'NONE':
                        relation.append((arg1_idx0, arg2_idx0, 0))
                    else:
                        relation.append((arg1_idx0, arg2_idx0, 1))
                        relation_type.append((arg1_idx0, arg2_idx0, relation_w2id[label]))
                    # negative same-as relation
                    sameas.append((arg1_idx0, arg2_idx0, 0))
                    sameas.append((arg2_idx0, arg1_idx0, 0))
            except:
                pass
        return sameas, relation, relation_type

    def preprocess(self, procset, relations_positive_negative=False):
        inputs = []
        for row in procset:
            text = row['text']
            tokens = row['tokens']

            size = len(tokens)
            entity = size * [0]
            multiword = []

            # entity and multiword gold-standard
            for keyid in row['keyphrases']:
                try:
                    keyphrase = row['keyphrases'][keyid]
                    idxs = keyphrase['idxs']

                    # mark only first subword with entity type
                    label_idx = entity_w2id[keyphrase['label']]
                    for idx in idxs:
                        if not tokens[idx].startswith('##'):
                            entity[idx] = label_idx

                    # multiword
                    for idx in idxs:
                        multiword.extend([(idx, idx_, 1) for idx_ in idxs if idx != idx_])

                    # negative examples (before first idx, after last ids,)
                    # - negative examples surrounding first and last idxs
                    idxs = sorted(idxs)
                    for idx in idxs:
                        for idx_ in range(size):
                            if idx_ not in idxs and entity[idx_] > 0:
                                multiword.append((idx, idx_, 0))
                                multiword.append((idx_, idx, 0))
                except:
                    pass

            # relations gold-standards
            if relations_positive_negative:
                sameas, relation, relation_type = self._get_relations_positive_negative_data(row)
            else:
                sameas, relation, relation_type = self._get_relations_data(row)

            inputs.append({
                'X': text,
                'entity': entity,
                'relation': relation,
                'relation_type': relation_type
            })
        return ProcDataset(inputs)

    def compute_loss_full(self, entity_probs, batch_entity, related_probs, batch_relation, related_type_probs,
                          batch_relation_type):
        # entity loss
        batch, seq_len, dim = entity_probs.size()
        entity_real = torch.nn.utils.rnn.pad_sequence(batch_entity).transpose(0, 1).to(self.device)
        entity_loss = self.criterion(entity_probs.view(batch*seq_len, dim), entity_real.view(-1))

        # relation loss
        batch, seq_len, dim = related_probs.size()
        rowcol_len = int(np.sqrt(seq_len))
        relation_real = torch.zeros((batch, rowcol_len, rowcol_len)).long().to(self.device)
        for i in range(batch):
            try:
                rows, columns = batch_relation[i][:, 0], batch_relation[i][:, 1]
                labels = batch_relation[i][:, 2]
                relation_real[i, rows, columns] = labels.to(self.device)
            except:
                pass

        relation_loss = self.criterion(related_probs.view(batch*seq_len, dim), relation_real.view(-1))

        batch, seq_len, dim = related_type_probs.size()
        rowcol_len = int(np.sqrt(seq_len))
        related_type_probs = related_type_probs.view((batch, rowcol_len, rowcol_len, dim))

        relation_real = []
        relation_pred = []

        for i in range(batch):
            try:
                rows, columns = batch_relation_type[i][:, 0], batch_relation_type[i][:, 1]
                labels = batch_relation_type[i][:, 2]
                relation_real.extend(labels.tolist())

                preds = related_type_probs[i, rows, columns]
                relation_pred.append(preds)
            except:
                pass

        try:
            relation_pred = torch.cat(relation_pred, 0).to(self.device)
            relation_real = torch.tensor(relation_real).to(self.device)
            relation_type_loss = self.criterion(relation_pred, relation_real)
        except:
            relation_type_loss = 0

        loss = entity_loss + relation_loss + relation_type_loss
        return loss

    def compute_loss(self, entity_probs, batch_entity, related_probs, batch_relation, related_type_probs,
                     batch_relation_type):
        # entity loss
        batch, seq_len, dim = entity_probs.size()
        entity_real = torch.nn.utils.rnn.pad_sequence(batch_entity).transpose(0, 1).to(self.device)
        entity_loss = self.criterion(entity_probs.view(batch * seq_len, dim), entity_real.reshape(-1))

        # relation loss
        batch, seq_len, dim = related_probs.size()
        rowcol_len = int(np.sqrt(seq_len))
        related_probs = related_probs.view((batch, rowcol_len, rowcol_len, dim))

        related_real = []
        related_pred = []
        for i in range(batch):
            try:
                rows, columns = batch_relation[i][:, 0], batch_relation[i][:, 1]
                labels = batch_relation[i][:, 2]
                related_real.extend(labels.tolist())

                preds = related_probs[i, rows, columns]
                related_pred.append(preds)
            except:
                pass

        try:
            related_pred = torch.cat(related_pred, 0).to(self.device)
            related_real = torch.tensor(related_real).to(self.device)
            relation_loss = self.criterion(related_pred, related_real)
        except:
            relation_loss = 0

        # relation type loss
        batch, seq_len, dim = related_type_probs.size()
        rowcol_len = int(np.sqrt(seq_len))
        related_type_probs = related_type_probs.view((batch, rowcol_len, rowcol_len, dim))

        relation_real = []
        relation_pred = []

        for i in range(batch):
            try:
                rows, columns = batch_relation_type[i][:, 0], batch_relation_type[i][:, 1]
                labels = batch_relation_type[i][:, 2]
                relation_real.extend(labels.tolist())

                preds = related_type_probs[i, rows, columns]
                relation_pred.append(preds)
            except:
                pass

        try:
            relation_pred = torch.cat(relation_pred, 0).to(self.device)
            relation_real = torch.tensor(relation_real).to(self.device)
            relation_type_loss = self.criterion(relation_pred, relation_real)
        except:
            relation_type_loss = 0

        loss = entity_loss + relation_loss + relation_type_loss
        return loss

    def train(self):
        # max_f1_score = self.eval()
        max_f1_score = 0
        repeat = 0
        for epoch in range(self.epochs):
            self.model.train()
            losses = []
            batch_X, batch_entity, batch_relation, batch_relation_type = [], [], [], []

            for batch_idx, inp in enumerate(self.traindata):                
                batch_X = inp['X']
                # Predict
                entity_probs, related_probs, related_type_probs = self.model(batch_X)

                self.optimizer.zero_grad()

                # Calculate loss
                batch_entity = torch.tensor([inp['entity']])
                batch_relation = torch.tensor([inp['relation']])
                batch_relation_type  = torch.tensor([inp['relation_type']])
                loss = self.compute_loss_full(entity_probs,
                                        batch_entity,
                                        related_probs,
                                        batch_relation,
                                        related_type_probs,
                                        batch_relation_type)
                losses.append(float(loss))

                # Backpropagation
                loss.backward()
                self.optimizer.step()

                # Display
                if (batch_idx + 1) % self.batch_status == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTotal Loss: {:.6f}'.format(
                        epoch, batch_idx + 1, len(self.traindata),
                               100. * batch_idx / len(self.traindata), float(loss), round(sum(losses) / len(losses), 5)))

            # evaluating
            self.eval_class_report()
            result_file_name = os.path.join(self.log_path, 'epoch' + str(epoch+1) + '.log')
            f1_score = self.eval(result_file_name)
            print('F1 score:', f1_score)
            if f1_score > max_f1_score:
                max_f1_score = f1_score
                repeat = 0

                print('Saving best model...')
                torch.save(self.model, self.write_path)
                print('Saving best model log...')
                best_log_path = os.path.join(self.log_path, 'best.log')
                copyfile(result_file_name, best_log_path)
            else:
                repeat += 1

            if repeat == self.early_stop:
                print('Total epochs:', (epoch + 1))
                break

    def eval_class_report(self, devdata=None):
        def _get_single_output_id_list(y):
            return [index for indexes in y for index in indexes]

        if devdata is not None:
            self.devdata = DataLoader(self.preprocess(devdata), batch_size=self.batch_size, shuffle=True)

        self.model.eval()

        entity_pred, entity_true, is_related_pred, is_related_true, relation_pred, relation_true = [], [], [], [], [], []
        for inp in self.devdata:
            sentence = inp['X']
            entity_ids = inp['entity']
            relation_ids = inp['relation']
            relation_type_ids  = inp['relation_type']
            
            # Predict
            entity_probs, related_probs, related_type_probs = self.model(sentence)

            len_sentence = entity_probs.shape[1]

            # Entity
            entity_pred.append([int(w) for w in list(entity_probs[0].argmax(dim=1))])
            entity_true.append([int(w) for w in list(entity_ids)])

            # Is Related
            related_array = [int(w) for w in list(related_probs[0].argmax(dim=1))]
            related_matrix = np.array(related_array).reshape((len_sentence, len_sentence))
            current_is_related_true, current_is_related_pred = self._get_relation_eval(relation_ids, related_matrix)
            is_related_true.extend(current_is_related_true)
            is_related_pred.extend(current_is_related_pred)

            # Relation type
            relation_type_array = [int(w) for w in list(related_type_probs[0].argmax(dim=1))]
            relation_type_matrix = np.array(relation_type_array).reshape((len_sentence, len_sentence))
            relation_type_true, relation_type_pred = self._get_relation_eval(relation_type_ids, relation_type_matrix)
            relation_true.extend(relation_type_true)
            relation_pred.extend(relation_type_pred)

        entity_labels = list(range(1, len(ENTITIES)))
        entity_target_names = ENTITIES[1:]
        print("Entity report:")
        print(classification_report(_get_single_output_id_list(entity_true), _get_single_output_id_list(entity_pred),
                                    labels=entity_labels, target_names=entity_target_names))
        print()

        print("Is related report:")
        print(classification_report(is_related_true, is_related_pred))
        print()

        relation_labels = list(range(len(utils.RELATIONS)))
        relation_target_names = utils.RELATIONS
        print("Relation type report")
        print(classification_report(relation_true, relation_pred, labels=relation_labels,
                                    target_names=relation_target_names))
        print()

    def test(self, devdata=None):

        if devdata is not None:
            test_dev_X = self.preprocess(devdata)
        else:
            test_dev_X = self.devdata

        self.model.eval()

        entity_pred, related_pred, relation_type_pred = [], [], []
        for sentence in test_dev_X:
            # Predict
            entity_probs, related_probs, related_type_probs = self.model(sentence['X'])

            len_sentence = entity_probs.shape[1]

            # Entity
            entity_pred.append([int(w) for w in list(entity_probs[0].argmax(dim=1))])

            # Related
            related_array = [int(w) for w in list(related_probs[0].argmax(dim=1))]
            related_matrix = np.array(related_array).reshape((len_sentence, len_sentence))
            related_pred.append(self._get_relation_output(related_matrix, 1, len_sentence))

            # Relation type
            relation_array = [int(w) for w in list(related_type_probs[0].argmax(dim=1))]
            relation_matrix = np.array(relation_array).reshape((len_sentence, len_sentence))
            relation_type_pred.append(self._get_relation_output(relation_matrix, 0, len_sentence))

        return entity_pred, related_pred, relation_type_pred

    def eval(self, result_file_name='result.txt', verbose=False, scenario=None):
        # mode = training | develop
        # pretrained_model = beto | multilingual

        devdata_folder = 'data/original/eval/%s/' % self.eval_mode
        output_folder = 'output/model/%s/' % self.eval_mode
        scenario_folder_list = ['scenario1-main/', 'scenario2-taskA/', 'scenario3-taskB/']
        for scenario_folder in scenario_folder_list:
            devdata_path = devdata_folder + scenario_folder + 'input_%s.json' % self.pretrained_model
            devdata = json.load(open(devdata_path))
            # devdata = DataLoader(self.preprocess(devdata), batch_size=self.batch_size, shuffle=True)
            entity_pred, related_pred, relation_pred = self.test(devdata)

            output_path = '%s/run1/%s/' % (output_folder, scenario_folder)
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            c = get_collection(devdata, entity_pred, related_pred, relation_pred)
            output_file_name = output_path + 'output.txt'
            c.dump(Path(output_file_name))
        command_text = "python3 data/scripts/score.py --gold {0} --submit {1}".format(devdata_folder, output_folder)
        if verbose:
            command_text += " --verbose"
        if scenario is not None:
            command_text += " --scenarios {0}".format(scenario)
        command_text += " > {0}".format(result_file_name)
        os.system(command_text)
        f1_score = 0.0
        with open(result_file_name, 'r') as f:
            print('Evaluation:')
            print(f.read())
        with open(result_file_name, 'r') as f:
            for line in f:
                if line.startswith("f1:"):
                    f1_score = float(line.split(':')[-1].strip())
                    break
        return f1_score

    @staticmethod
    def _get_relation_output(relation_matrix, relation_value, len_sentence):
        relation_output = []
        for i in range(len_sentence):
            for j in range(len_sentence):
                if relation_matrix[i, j] >= relation_value:
                    relation_output.append((i, j, int(relation_matrix[i, j])))
        return relation_output

    @staticmethod
    def _get_relation_eval(relation_ids, relation_matrix):
        relation_true, relation_pred = [], []
        for relation in relation_ids:
            idx1, idx2, label = relation
            relation_true.append(int(label))
            relation_pred.append(int(relation_matrix[idx1, idx2]))
            # relation_output.append((int(idx1), int(idx2), int(relation_matrix[idx1, idx2])))
        return relation_true, relation_pred