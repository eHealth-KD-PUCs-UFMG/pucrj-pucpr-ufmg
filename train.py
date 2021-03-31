
import torch
import json
import torch
import torch.nn as nn
from torch import optim
from model import Vicomtech
import numpy as np
from sklearn.metrics import classification_report
import utils

class Train:
    def __init__(self, model, criterion, optimizer, traindata, devdata, epochs, batch_size, batch_status=16,
                 early_stop=5, device='cuda'):
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

        self.train_X, self.train_entity, self.train_multiword, self.train_sameas, self.train_relation, self.train_relation_type = self.preprocess(
            traindata)
        self.dev_X, self.dev_entity, self.dev_multiword, self.dev_sameas, self.dev_relation, self.dev_relation_type = self.preprocess(
            devdata)

    def preprocess(self, procset):
        X, y_entity, y_multiword, y_sameas, y_relation, y_relation_type = [], [], [], [], [], []
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
                    label_idx = utils.entity_w2id[keyphrase['label']]
                    for idx in idxs:
                        if not tokens[idx].startswith('##'):
                            entity[idx] = label_idx

                    # multiword
                    for idx in idxs:
                        multiword.extend([(idx, idx_, 1) for idx_ in idxs if idx != idx_])

                    # negative examples (before first idx, after last ids,)
                    # - negative examples surrounding first and last idxs
                    idxs = sorted(idxs)
                    end = idxs[-1]
                    for idx in idxs:
                        for idx_ in range(idx, end):
                            if idx_ not in idxs:
                                multiword.append((idx, idx_, 0))
                    # - negative examples between first idx and its 3 neighboor antecedents
                    for start in range(max(0, idxs[0]-3), idxs[0]):
                        multiword.append((start, idxs[0], 0))
                    # - negative examples between last idx and its 3 neighboor successor
                    for end in range(idxs[-1]+1, min(idxs[-1]+3, len(tokens))):
                        multiword.append((idxs[-1], end, 0))
                except:
                    pass

            # relations gold-standards
            sameas, relation, relation_type = [], [], []
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
                        relation_type.append((arg1_idx0, arg2_idx0, utils.relation_w2id[label]))
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

            X.append(text)
            y_entity.append(torch.tensor(entity))
            y_multiword.append(torch.tensor(multiword))
            y_sameas.append(torch.tensor(sameas))
            y_relation.append(torch.tensor(relation))
            y_relation_type.append(torch.tensor(relation_type))
        return X, y_entity, y_multiword, y_sameas, y_relation, y_relation_type

    def compute_loss(self, entity_probs, batch_entity, multiword_probs, batch_multiword, \
                     sameas_probs, batch_sameas, related_probs, batch_relation, \
                     related_type_probs, batch_relation_type):
        # entity loss
        batch, seq_len, dim = entity_probs.size()
        entity_real = torch.nn.utils.rnn.pad_sequence(batch_entity).transpose(0, 1).to(self.device)
        entity_loss = self.criterion(entity_probs.view(batch * seq_len, dim), entity_real.reshape(-1))

        # multiword loss
        batch, seq_len, dim = multiword_probs.size()
        rowcol_len = int(np.sqrt(seq_len))
        multiword_probs = multiword_probs.view((batch, rowcol_len, rowcol_len, dim))

        multiword_real = []
        multiword_pred = []
        for i in range(batch):
            try:
                rows, columns = batch_multiword[i][:, 0], batch_multiword[i][:, 1]
                labels = batch_multiword[i][:, 2]
                multiword_real.extend(labels.tolist())

                preds = multiword_probs[i, rows, columns]
                multiword_pred.append(preds)
            except:
                pass

        try:
            multiword_pred = torch.cat(multiword_pred, 0).to(self.device)
            multiword_real = torch.tensor(multiword_real).to(self.device)
            multiword_loss = self.criterion(multiword_pred, multiword_real)
        except:
            multiword_loss = 0

        # sameas loss
        batch, seq_len, dim = sameas_probs.size()
        rowcol_len = int(np.sqrt(seq_len))
        sameas_probs = sameas_probs.view((batch, rowcol_len, rowcol_len, dim))

        sameas_real = []
        sameas_pred = []
        for i in range(batch):
            try:
                rows, columns = batch_sameas[i][:, 0], batch_sameas[i][:, 1]
                labels = batch_sameas[i][:, 2]
                sameas_real.extend(labels.tolist())

                preds = sameas_probs[i, rows, columns]
                sameas_pred.append(preds)
            except:
                pass

        try:
            sameas_pred = torch.cat(sameas_pred, 0).to(self.device)
            sameas_real = torch.tensor(sameas_real).to(self.device)
            sameas_loss = self.criterion(sameas_pred, sameas_real)
        except:
            sameas_loss = 0

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

        loss = entity_loss + multiword_loss + sameas_loss + relation_loss + relation_type_loss
        return loss

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            losses = []
            batch_X, batch_entity, batch_multiword, batch_sameas, batch_relation, batch_relation_type = [], [], [], [], [], []

            for batch_idx, inp in enumerate(self.train_X):
                batch_X.append(self.train_X[batch_idx])
                batch_entity.append(self.train_entity[batch_idx])
                batch_multiword.append(self.train_multiword[batch_idx])
                batch_sameas.append(self.train_sameas[batch_idx])
                batch_relation.append(self.train_relation[batch_idx])
                batch_relation_type.append(self.train_relation_type[batch_idx])

                if (batch_idx + 1) % self.batch_size == 0:
                    # Init
                    self.optimizer.zero_grad()

                    # Predict
                    entity_probs, multiword_probs, sameas_probs, related_probs, related_type_probs = self.model(batch_X)

                    # Calculate loss
                    loss = self.compute_loss(entity_probs,
                                             batch_entity,
                                             multiword_probs,
                                             batch_multiword,
                                             sameas_probs, batch_sameas,
                                             related_probs,
                                             batch_relation,
                                             related_type_probs,
                                             batch_relation_type)
                    losses.append(float(loss))

                    # Backpropagation
                    loss.backward()
                    self.optimizer.step()

                    batch_X, batch_entity, batch_multiword, batch_sameas, batch_relation, batch_relation_type = [], [], [], [], [], []

                # Display
                if (batch_idx + 1) % self.batch_status == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTotal Loss: {:.6f}'.format(
                        epoch, batch_idx + 1, len(self.train_X),
                               100. * batch_idx / len(self.train_X), float(loss), round(sum(losses) / len(losses), 5)))
        
            # evaluating
            self.eval()

    def eval(self):
        def _get_single_output_id_list(y):
            return [index for indexes in y for index in indexes]

        self.model.eval()

        entity_pred, entity_true, is_related_pred, is_related_true, multiword_pred, multiword_true, relation_pred, relation_true = [], [], [], [], [], [], [], []
        for sentence, entity_ids, multiword_ids, relation_ids, relation_type_ids in zip(self.dev_X, self.dev_entity,
                                                                                        self.dev_multiword,
                                                                                        self.dev_relation,
                                                                                        self.dev_relation_type):
            # Predict
            entity_probs, multiword_probs, sameas_probs, related_probs, related_type_probs = self.model(sentence)

            len_sentence = entity_probs.shape[1]

            # Entity
            entity_pred.append([int(w) for w in list(entity_probs[0].argmax(dim=1))])
            entity_true.append([int(w) for w in list(entity_ids)])

            # Multiword
            multiword_array = [int(w) for w in list(multiword_probs[0].argmax(dim=1))]
            multiword_matrix = np.array(multiword_array).reshape((len_sentence, len_sentence))
            current_multiword_true, current_multiword_pred = self._get_relation_eval(multiword_ids, multiword_matrix)
            multiword_true.extend(current_multiword_true)
            multiword_pred.extend(current_multiword_pred)

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
            for relation in relation_ids:
                idx1, idx2, label = relation
                if label == 0:
                    relation_type_true.append(int(label))
                    relation_type_pred.append(int(relation_type_matrix[idx1, idx2]))
            relation_true.extend(relation_type_true)
            relation_pred.extend(relation_type_pred)

        entity_labels = list(range(1, len(utils.ENTITIES)))
        entity_target_names = utils.ENTITIES[1:]
        print("Entity report:")
        print(classification_report(_get_single_output_id_list(entity_true), _get_single_output_id_list(entity_pred),
                                    labels=entity_labels, target_names=entity_target_names))
        print()

        print("Multiword report:")
        print(classification_report(multiword_true, multiword_pred))
        print()

        print("Is related report:")
        print(classification_report(is_related_true, is_related_pred))
        print()

        relation_labels = list(range(1, len(utils.RELATIONS)))
        relation_target_names = utils.RELATIONS[1:]
        print("Relation type report")
        print(classification_report(relation_true, relation_pred, labels=relation_labels,
                                    target_names=relation_target_names))
        print()

    def test(self):
        self.model.eval()

        entity_pred, multiword_pred, sameas_pred, related_pred, relation_type_pred = [], [], [], [], []
        for sentence in self.dev_X:
            # Predict
            entity_probs, multiword_probs, sameas_probs, related_probs, related_type_probs = self.model(sentence)

            len_sentence = entity_probs.shape[1]

            # Entity
            entity_pred.append([int(w) for w in list(entity_probs[0].argmax(dim=1))])

            # Multiword
            multiword_array = [int(w) for w in list(multiword_probs[0].argmax(dim=1))]
            multiword_matrix = np.array(multiword_array).reshape((len_sentence, len_sentence))
            multiword_pred.append(self._get_relation_output(multiword_matrix, 1, len_sentence))

            # Same-as
            sameas_array = [int(w) for w in list(sameas_probs[0].argmax(dim=1))]
            sameas_matrix = np.array(sameas_array).reshape((len_sentence, len_sentence))
            sameas_pred.append(self._get_relation_output(sameas_matrix, 1, len_sentence))

            # Related
            related_array = [int(w) for w in list(related_probs[0].argmax(dim=1))]
            related_matrix = np.array(related_array).reshape((len_sentence, len_sentence))
            related_pred.append(self._get_relation_output(related_matrix, 1, len_sentence))

            # Relation type
            relation_array = [int(w) for w in list(related_type_probs[0].argmax(dim=1))]
            relation_matrix = np.array(relation_array).reshape((len_sentence, len_sentence))
            relation_type_pred.append(self._get_relation_output(relation_matrix, 1, len_sentence))

        return entity_pred, multiword_pred, sameas_pred, related_pred, relation_type_pred

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