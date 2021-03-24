
import torch
import json
import torch
import torch.nn as nn
from torch import optim
from model import Vicomtech
import numpy as np
from sklearn.metrics import classification_report


ENTITIES = ["O", "Concept", "Action", "Predicate", "Reference"]

RELATIONS = [
    "O",
    "is-a",
    "part-of",
    "has-property",
    "causes",
    "entails",
    "in-context",
    "in-place",
    "in-time",
    "subject",
    "target",
    "domain",
    "arg",
]

entity_w2id = { w:i for i, w in enumerate(ENTITIES) }
entity_id2w = { i:w for i, w in enumerate(ENTITIES) }
relation_w2id = { w:i for i, w in enumerate(RELATIONS) }
relation_id2w = { i:w for i, w in enumerate(RELATIONS) }

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
        self.dev_X, self.dev_entity, self.dev_multiword, self.dev_sameas, self.dev_relation, self.dev_relation_type = self.preprocess(devdata)

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
                    end = idxs[-1]
                    for idx in idxs:
                        for idx_ in range(idx, end):
                            if idx_ not in idxs:
                                multiword.append((idx, idx_, 0))
                    # - negative examples between first idx and its neighboor antecedent
                    start = idxs[0]
                    if start > 0:
                        multiword.append((start-1, start, 0))
                    # - negative examples between last idx and its neighboor successor
                    end = idxs[-1]
                    if end+1 < len(idxs):
                        multiword.append((end, end+1, 0))
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
        self.model.train()

        for epoch in range(self.epochs):
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


    def eval(self):
        def get_single_output_id_list(y):
            return [index for indexes in y for index in indexes]

        self.model.eval()

        entity_pred, entity_true, is_related_pred, is_related_true, multiword_pred, multiword_pred_output, multiword_true, relation_pred, relation_true = [], [], [], [], [], [], [], [], []
        for sentence, entity_ids, multiword_ids, relation_ids in zip(self.dev_X, self.dev_entity, self.dev_multiword,
                                                                     self.dev_relation):
            # Predict
            entity_probs, multiword_probs, sameas_probs, related_probs, related_type_probs = self.model(sentence)

            len_sentence = entity_probs.shape[1]

            # Entity
            entity_pred.append([int(w) for w in list(entity_probs[0].argmax(dim=1))])
            entity_true.append([int(w) for w in list(entity_ids)])

            # Multiword
            current_multiword_pred = [int(w) for w in list(multiword_probs[0].argmax(dim=1))]
            multiword_matrix = np.array(current_multiword_pred).reshape((len_sentence, len_sentence))
            multiword_output = []
            for i in range(len_sentence):
                for j in range(len_sentence):
                    if multiword_matrix[i, j] == 1:
                        multiword_output.append((i, j))
            multiword_pred_output.append(multiword_output)
            multiword_pred.extend(current_multiword_pred)
            current_multiword_true = np.zeros((len_sentence, len_sentence))
            for multiword in multiword_ids:
                idx1, idx2 = multiword[0], multiword[1]
                current_multiword_true[idx1, idx2] = 1
            current_multiword_true = current_multiword_true.reshape(len_sentence ** 2)
            multiword_true.extend([int(w) for w in list(current_multiword_true)])

            # Is Related
            current_is_related_pred = [int(w) for w in list(related_probs[0].argmax(dim=1))]
            is_related_pred.extend(current_is_related_pred)
            current_is_related_true = np.zeros((len_sentence, len_sentence))
            for relation in relation_ids:
                idx1, idx2 = relation[0], relation[1]
                current_is_related_true[idx1, idx2] = 1
            current_is_related_true = current_is_related_true.reshape(len_sentence ** 2)
            is_related_true.extend([int(w) for w in list(current_is_related_true)])

            # Relation type
            current_relation_pred = [int(w) for w in list(related_type_probs[0].argmax(dim=1))]
            relation_pred.extend(current_relation_pred)
            current_relation_true = np.zeros((len_sentence, len_sentence))
            for relation in relation_ids:
                idx1, idx2, label = relation[0], relation[1], relation[2]
                current_relation_true[idx1, idx2] = label
            current_relation_true = current_relation_true.reshape(len_sentence ** 2)
            relation_true.extend([int(w) for w in list(current_relation_true)])

        entity_labels = list(range(1, len(ENTITIES)))
        entity_target_names = ENTITIES[1:]
        print("Entity report:")
        print(classification_report(get_single_output_id_list(entity_true), get_single_output_id_list(entity_pred),
                                    labels=entity_labels, target_names=entity_target_names))
        print()

        print("Is related report:")
        print(classification_report(multiword_true, multiword_pred))
        print()

        print("Multiword report:")
        print(classification_report(is_related_true, is_related_pred))
        print()

        relation_labels = list(range(0, len(RELATIONS)))
        relation_target_names = RELATIONS[0:]
        print("Relation type report")
        print(classification_report(relation_true, relation_pred, labels=relation_labels,
                                    target_names=relation_target_names))
        print()
        return entity_pred, entity_true, multiword_pred_output, is_related_pred, relation_pred