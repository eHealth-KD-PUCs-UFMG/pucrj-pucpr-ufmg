
import json
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import classification_report
import os
from shutil import copyfile
from pathlib import Path

import utils
import postprocessing

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
    def __init__(self, model, criterion, optimizer, loss_func, loss_optimizer, traindata, devdata, epochs, batch_size, batch_status=16,
                 early_stop=3, device='cuda', write_path='model.pt', eval_mode='develop',
                 pretrained_model='multilingual', log_path='logs', relations_positive_negative=False, relations_inv=False):
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

        self.loss_func = loss_func
        self.loss_optimizer = loss_optimizer

        self.write_path = write_path
        self.log_path = log_path
        if not os.path.exists(log_path):
            os.mkdir(log_path)

        self.eval_mode = eval_mode
        self.pretrained_model = pretrained_model

        self.relations_inv = relations_inv
        self.entities = utils.ENTITIES
        self.relations = utils.RELATIONS
        self.entity_w2id = utils.entity_w2id
        self.relation_w2id = utils.relation_w2id
        if relations_inv:
            self.relations = utils.RELATIONS_INV
            self.relation_w2id = utils.relation_inv_w2id

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

    def _get_relations_data(self, row):
        relation, relation_type = [], []
        for relation_ in row['relations']:
            try:
                arg1 = relation_['arg1']
                arg1_idx0 = row['keyphrases'][str(arg1)]['idxs'][0]

                arg2 = relation_['arg2']
                arg2_idx0 = row['keyphrases'][str(arg2)]['idxs'][0]

                label = relation_['label']
                relation.append((arg1_idx0, arg2_idx0, self.relation_w2id[label]))
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

                f = [w for w in relation if w[0] == arg2_idx0 and w[1] == arg1_idx0]
                if len(f) == 0:
                    relation.append((arg2_idx0, arg1_idx0, 0))
        return relation#, relation_type


    def _get_relations_positive_negative_data(self, row):
        relation, relation_type = [], []
        for relation_ in row['relations_positive_negative']:
            try:
                arg1 = relation_['arg1']
                arg1_idx0 = row['keyphrases'][str(arg1)]['idxs'][0]

                arg2 = relation_['arg2']
                arg2_idx0 = row['keyphrases'][str(arg2)]['idxs'][0]

                label = relation_['label']
                if label == 'NONE':
                    relation.append((arg1_idx0, arg2_idx0, 0))
                else:
                    relation.append((arg1_idx0, arg2_idx0, self.relation_w2id[label]))
            except:
                pass
        return relation#, relation_type

    def preprocess(self, procset, relations_positive_negative=False):
        inputs = []
        for row in procset:
            text = row['text']
            tokens = row['tokens']

            size = len(tokens)
            entity = size * [0]

            # entity gold-standard
            for keyid in row['keyphrases']:
                try:
                    keyphrase = row['keyphrases'][keyid]
                    idxs = keyphrase['idxs']

                    # mark only first subword with entity type
                    label_begin_idx = self.entity_w2id['B-' + keyphrase['label']]
                    label_internal_idx = self.entity_w2id['I-' + keyphrase['label']]
                    first = True
                    for idx in idxs:
                        if not tokens[idx].startswith('##'):
                            if first:
                                entity[idx] = label_begin_idx
                                first = False
                            else:
                                entity[idx] = label_internal_idx
                except:
                    pass

            # relations gold-standards
            if relations_positive_negative:
                relation = self._get_relations_positive_negative_data(row)
            else:
                relation = self._get_relations_data(row)

            inputs.append({
                'X': text,
                'entity': entity,
                'relation': relation,
            })
        return ProcDataset(inputs)

    def compute_loss_full(self, entity_probs, batch_entity, related_probs, batch_relation):
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

        loss = self.loss_func(entity_loss, relation_loss)
        return loss

    def compute_loss(self, entity_probs, batch_entity, related_probs, batch_relation):
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

        loss = self.loss_func(entity_loss, relation_loss)
        return loss

    def train(self):
        max_f1_score = self.eval()
        repeat = 0
        for epoch in range(self.epochs):
            self.model.train()
            losses = []

            for batch_idx, inp in enumerate(self.traindata):                
                batch_X = inp['X']
                # Predict
                entity_probs, related_probs = self.model(batch_X)

                self.optimizer.zero_grad()
                self.loss_optimizer.zero_grad()

                # Calculate loss
                batch_entity = torch.tensor([inp['entity']])
                batch_relation = torch.tensor([inp['relation']])
                loss = self.compute_loss_full(entity_probs,
                                        batch_entity,
                                        related_probs,
                                        batch_relation)
                losses.append(float(loss))

                # Backpropagation
                loss.backward()
                self.optimizer.step()
                self.loss_optimizer.step()

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
            
            # Predict
            entity_probs, related_probs = self.model(sentence)

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

        entity_labels = list(range(1, len(self.entities)))
        entity_target_names = self.entities[1:]
        print("Entity report:")
        print(classification_report(_get_single_output_id_list(entity_true), _get_single_output_id_list(entity_pred),
                                    labels=entity_labels, target_names=entity_target_names))
        print()

        relation_labels = list(range(len(self.relations)))
        relation_target_names = self.relations
        print("Is related report:")
        print(classification_report(is_related_true, is_related_pred, labels=relation_labels,
                                    target_names=relation_target_names))
        print()

    def test(self, devdata=None):

        if devdata is not None:
            test_dev_X = self.preprocess(devdata)
        else:
            test_dev_X = self.devdata

        self.model.eval()

        entity_pred, related_pred = [], []
        for sentence in test_dev_X:
            # Predict
            entity_probs, related_probs = self.model(sentence['X'])

            len_sentence = entity_probs.shape[1]

            # Entity
            entity_pred.append([int(w) for w in list(entity_probs[0].argmax(dim=1))])

            # Related
            related_array = [int(w) for w in list(related_probs[0].argmax(dim=1))]
            related_matrix = np.array(related_array).reshape((len_sentence, len_sentence))
            related_pred.append(self._get_relation_output(related_matrix, 1, len_sentence))

        return entity_pred, related_pred

    def eval(self, result_file_name='result.txt', verbose=False, scenario=None):
        # mode = training | develop
        # pretrained_model = beto | multilingual

        devdata_folder = 'data/original/eval/%s/' % self.eval_mode
        output_folder = 'output/model/%s/' % self.eval_mode
        scenario_folder_list = ['scenario1-main/', 'scenario2-taskA/', 'scenario3-taskB/']
        for scenario_folder in scenario_folder_list:
            devdata_path = devdata_folder + scenario_folder + 'input_%s.json' % self.pretrained_model
            devdata = json.load(open(devdata_path))
            entity_pred, related_pred = self.test(devdata)

            output_path = '%s/run1/%s/' % (output_folder, scenario_folder)
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            c = postprocessing.get_collection(devdata, entity_pred, related_pred, relations_inv=self.relations_inv)
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
        return relation_true, relation_pred