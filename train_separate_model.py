import json
import torch
import numpy as np
import os
from shutil import copyfile
from pathlib import Path

from utils import relation_id2w, relation_w2id, entity_id2w, entity_w2id
from postprocessing import get_collection

class TrainSeparateModel:
    def __init__(self, entity_model, relation_model, criterion, entity_optimizer, relation_optimizer, traindata, devdata,
                 epochs, batch_size, batch_status=16, early_stop=3, device='cuda', write_path='model.pt',
                 eval_mode='develop', pretrained_model='multilingual', log_path='logs'):
        self.epochs = epochs
        self.batch_size = batch_size
        self.batch_status = batch_status
        self.early_stop = early_stop
        self.device = device

        self.traindata = traindata
        self.devdata = devdata

        self.entity_model = entity_model
        self.relation_model = relation_model
        self.criterion = criterion
        self.entity_optimizer = entity_optimizer
        self.relation_optimizer = relation_optimizer

        self.write_path = write_path
        self.log_path = log_path
        if not os.path.exists(log_path):
            os.mkdir(log_path)

        self.eval_mode = eval_mode
        self.pretrained_model = pretrained_model

        self.train_X, self.train_entity, self.train_relation, self.train_relation_type = self.preprocess(
            traindata)
        self.dev_X, self.dev_entity, self.dev_relation, self.dev_relation_type = self.preprocess(
            devdata)

    def __str__(self):
        return "Epochs: {}\nBatch size: {}\nEarly stop: {}\nData: {}\nPretrained model: {}".format(self.epochs,
                                                                                                   self.batch_size,
                                                                                                   self.early_stop,
                                                                                                   self.eval_mode,
                                                                                                   self.pretrained_model)

    def preprocess(self, procset):
        X, y_entity, y_relation, y_relation_type = [], [], [], []
        for row in procset:
            text = row['text']
            tokens = row['tokens']

            size = len(tokens)
            entity = size * [0]

            # entity gold-standard
            entity_idxs = []
            for keyid in row['keyphrases']:
                try:
                    keyphrase = row['keyphrases'][keyid]
                    idxs = keyphrase['idxs']

                    if len(idxs) > 0:
                        entity_idxs.append(idxs[0])

                    # mark only first subword with entity type
                    label_begin_idx = entity_w2id['B-' + keyphrase['label']]
                    label_internal_idx = entity_w2id['I-' + keyphrase['label']]
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
            relation_pairs = []
            relation, relation_type = [], []
            for relation_ in row['relations']:
                try:
                    arg1 = relation_['arg1']
                    arg1_idx0 = row['keyphrases'][str(arg1)]['idxs'][0]

                    arg2 = relation_['arg2']
                    arg2_idx0 = row['keyphrases'][str(arg2)]['idxs'][0]
                    relation_pairs.append((arg1_idx0, arg2_idx0))

                    label = relation_['label']
                    relation.append((arg1_idx0, arg2_idx0, 1))
                    relation_type.append((arg1_idx0, arg2_idx0, relation_w2id[label]))
                except:
                    pass

            # negative relation examples
            for idx in entity_idxs:
                for idx_ in entity_idxs:
                    if idx != idx_ and not (idx, idx_) in relation_pairs:
                        relation.append((idx, idx_, 0))

            X.append(text)
            y_entity.append(torch.tensor(entity))
            y_relation.append(torch.tensor(relation))
            y_relation_type.append(torch.tensor(relation_type))
        return X, y_entity, y_relation, y_relation_type

    def compute_loss_entity(self, entity_probs, batch_entity):
        # entity loss
        batch, seq_len, dim = entity_probs.size()
        entity_real = torch.nn.utils.rnn.pad_sequence(batch_entity).transpose(0, 1).to(self.device)
        entity_loss = self.criterion(entity_probs.view(batch * seq_len, dim), entity_real.reshape(-1))
        return entity_loss

    def compute_loss_relation(self, related_probs, batch_relation, related_type_probs,
                              batch_relation_type):
        # multiword loss - removed multiword

        # sameas loss - removed sameas

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

        loss = relation_loss + relation_type_loss
        return loss

    def train(self):
        max_f1_score = self.eval()
        repeat = 0
        for epoch in range(self.epochs):
            self.entity_model.train()
            self.relation_model.train()
            losses_entity, losses_relation = [], []
            batch_X, batch_entity, batch_relation, batch_relation_type = [], [], [], []

            for batch_idx, inp in enumerate(self.train_X):
                batch_X.append(self.train_X[batch_idx])
                batch_entity.append(self.train_entity[batch_idx])
                batch_relation.append(self.train_relation[batch_idx])
                batch_relation_type.append(self.train_relation_type[batch_idx])

                if (batch_idx + 1) % self.batch_size == 0:
                    # Init
                    self.entity_optimizer.zero_grad()
                    self.relation_optimizer.zero_grad()

                    # Predict
                    entity_probs, embeddings = self.entity_model(batch_X)
                    related_probs, related_type_probs = self.relation_model(embeddings)

                    # Calculate loss
                    loss_entity = self.compute_loss_entity(entity_probs, batch_entity)
                    loss_relation = self.compute_loss_relation(related_probs, batch_relation, related_type_probs,
                                                               batch_relation_type)
                    losses_entity.append(float(loss_entity))
                    losses_relation.append(float(loss_relation))

                    # Backpropagation
                    loss_entity.backward(retain_graph=True)
                    loss_relation.backward()
                    self.entity_optimizer.step()
                    self.relation_optimizer.step()

                    batch_X, batch_entity, batch_relation, batch_relation_type = [], [], [], []

                # Display
                if (batch_idx + 1) % self.batch_status == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss Entity: {:.6f}\tTotal Loss Entity: {:.6f}'.format(
                        epoch, batch_idx + 1, len(self.train_X),
                               100. * batch_idx / len(self.train_X), float(loss_entity), round(sum(losses_entity) / len(losses_entity), 5)))
                    print('\t\t\tLoss Relation: {:.6f}\tTotal Loss Relation: {:.6f}'.format(
                        float(loss_relation),
                        round(sum(losses_relation) / len(losses_relation), 5)))
            # evaluating
            result_file_name = os.path.join(self.log_path, 'epoch' + str(epoch + 1) + '.log')
            f1_score = self.eval(result_file_name)
            print('F1 score:', f1_score)
            if f1_score > max_f1_score:
                max_f1_score = f1_score
                repeat = 0

                print('Saving best model...')
                torch.save(self.entity_model, 'entity_{}'.format(self.write_path))
                torch.save(self.relation_model, 'relation_{}'.format(self.write_path))
                print('Saving best model log...')
                best_log_path = os.path.join(self.log_path, 'best.log')
                copyfile(result_file_name, best_log_path)
            else:
                repeat += 1

            if repeat == self.early_stop:
                print('Total epochs:', (epoch + 1))
                break

    def eval(self, result_file_name='result.txt', verbose=False, scenario=None):
        # mode = training | develop
        # pretrained_model = beto | multilingual

        devdata_folder = 'data/original/eval/%s/' % self.eval_mode
        output_folder = 'output/model/%s/' % self.eval_mode
        scenario_folder_list = ['scenario1-main/', 'scenario2-taskA/', 'scenario3-taskB/']
        for scenario_folder in scenario_folder_list:
            devdata_path = devdata_folder + scenario_folder + 'input_%s.json' % self.pretrained_model
            devdata = json.load(open(devdata_path))
            entity_pred, related_pred, relation_pred = self.test(devdata)

            output_path = '%s/run1/%s/' % (output_folder, scenario_folder)
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            c = get_collection(devdata, entity_pred, related_pred, relation_pred)
            output_file_name = output_path + 'output.txt'
            c.dump(Path(output_file_name))
        command_text = "python data/scripts/score.py --gold {0} --submit {1}".format(devdata_folder, output_folder)
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

    def test(self, devdata=None):

        if devdata is not None:
            test_dev_X, _, _, _ = self.preprocess(devdata)
        else:
            test_dev_X = self.dev_X

        self.entity_model.eval()
        self.relation_model.eval()

        entity_pred, related_pred, relation_type_pred = [], [], []
        for sentence in test_dev_X:
            # Predict
            entity_probs, embeddings = self.entity_model(sentence)
            related_probs, related_type_probs = self.relation_model(embeddings)

            len_sentence = entity_probs.shape[1]

            # Entity
            entity_pred.append([int(w) for w in list(entity_probs[0].argmax(dim=1))])

            # Multiword - removed multiword

            # Related
            related_array = [int(w) for w in list(related_probs[0].argmax(dim=1))]
            related_matrix = np.array(related_array).reshape((len_sentence, len_sentence))
            related_pred.append(self._get_relation_output(related_matrix, 1, len_sentence))

            # Relation type
            relation_array = [int(w) for w in list(related_type_probs[0].argmax(dim=1))]
            relation_matrix = np.array(relation_array).reshape((len_sentence, len_sentence))
            relation_type_pred.append(self._get_relation_output(relation_matrix, 1, len_sentence))

        return entity_pred, related_pred, relation_type_pred

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