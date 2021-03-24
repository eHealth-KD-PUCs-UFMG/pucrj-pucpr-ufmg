
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
    def __init__(self, model, criterion, optimizer, traindata, devdata, epochs, batch_size, batch_status=16, early_stop=5, device='cuda'):
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

        self.train_X, self.train_entity, self.train_multiword, self.train_sameas, self.train_relation = self.preprocess(traindata)
        self.dev_X, self.dev_entity, self.dev_multiword, self.dev_sameas, self.dev_relation = self.preprocess(devdata)

    def preprocess(self, procset):
        X, y_entity, y_multiword, y_sameas, y_relation = [], [], [], [], []
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
                    entity[idxs[0]] = label_idx
                    
                    # multiword
                    for idx in idxs:
                        multiword.extend([(idx, idx_) for idx_ in idxs])
                except:
                    pass

            # relations gold-standards
            sameas, relation = [], []
            for relation_ in row['relations']:
                try:
                    arg1 = relation_['arg1']
                    arg1_idx0 = row['keyphrases'][str(arg1)]['idxs'][0]

                    arg2 = relation_['arg2']
                    arg2_idx0 = row['keyphrases'][str(arg2)]['idxs'][0]

                    label = relation_['label']
                    if label == 'same-as':
                        sameas.append((arg1_idx0, arg2_idx0))
                    else:
                        relation.append((arg1_idx0, arg2_idx0, relation_w2id[label]))
                except:
                    pass
            
            X.append(text)
            y_entity.append(torch.tensor(entity))
            y_multiword.append(torch.tensor(multiword))
            y_sameas.append(torch.tensor(sameas))
            y_relation.append(torch.tensor(relation))
        return X, y_entity, y_multiword, y_sameas, y_relation


    def compute_loss(self, entity_probs, batch_entity, multiword_probs, batch_multiword, \
                    sameas_probs, batch_sameas, related_probs, batch_relation,\
                    related_type_probs):
        # entity loss
        batch, seq_len, dim = entity_probs.size()
        entity_real = torch.nn.utils.rnn.pad_sequence(batch_entity).transpose(0, 1).to(self.device)
        entity_loss = self.criterion(entity_probs.view(batch*seq_len, dim), entity_real.reshape(-1))

        # multiword loss
        batch, seq_len, dim = multiword_probs.size()
        rowcol_len = int(np.sqrt(seq_len))
        multiword_real = torch.zeros((batch, rowcol_len, rowcol_len)).long().to(self.device)
        for i in range(batch):
            rows, columns = batch_multiword[i][:, 0], batch_multiword[i][:, 1]
            multiword_real[i, rows, columns] = 1

        multiword_loss = self.criterion(multiword_probs.view(batch*seq_len, dim), multiword_real.view(-1))

        # sameas loss
        batch, seq_len, dim = sameas_probs.size()
        rowcol_len = int(np.sqrt(seq_len))
        sameas_real = torch.zeros((batch, rowcol_len, rowcol_len)).long().to(self.device)
        for i in range(batch):
            try:
                rows, columns = batch_sameas[i][:, 0], batch_sameas[i][:, 1]
                sameas_real[i, rows, columns] = 1
            except:
                pass

        sameas_loss = self.criterion(sameas_probs.view(batch*seq_len, dim), sameas_real.view(-1))

        # relation loss
        batch, seq_len, dim = related_probs.size()
        rowcol_len = int(np.sqrt(seq_len))
        relation_real = torch.zeros((batch, rowcol_len, rowcol_len)).long().to(self.device)
        for i in range(batch):
            try:
                rows, columns = batch_relation[i][:, 0], batch_relation[i][:, 1]
                relation_real[i, rows, columns] = 1
            except:
                pass

        relation_loss = self.criterion(related_probs.view(batch*seq_len, dim), relation_real.view(-1))

        # relation type loss
        batch, seq_len, dim = related_type_probs.size()
        rowcol_len = int(np.sqrt(seq_len))
        relation_real = torch.zeros((batch, rowcol_len, rowcol_len)).long().to(self.device)
        for i in range(batch):
            try:
                rows, columns = batch_relation[i][:, 0], batch_relation[i][:, 1]
                labels = batch_relation[i][:, 2]
                relation_real[i, rows, columns] = labels
            except:
                pass

        relation_type_loss = self.criterion(related_type_probs.view(batch*seq_len, dim), relation_real.view(-1))

        loss = entity_loss + multiword_loss + sameas_loss + relation_loss + relation_type_loss
        return loss


    def train(self):
        self.model.train()

        for epoch in range(self.epochs):
            losses = []
            batch_X, batch_entity, batch_multiword, batch_sameas, batch_relation = [], [], [], [], []

            for batch_idx, inp in enumerate(self.train_X):
                batch_X.append(self.train_X[batch_idx])
                batch_entity.append(self.train_entity[batch_idx])
                batch_multiword.append(self.train_multiword[batch_idx])
                batch_sameas.append(self.train_sameas[batch_idx])
                batch_relation.append(self.train_relation[batch_idx])

                if (batch_idx+1) % self.batch_size == 0:
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
                                             related_type_probs)
                    losses.append(float(loss))

                    # Backpropagation
                    loss.backward()
                    self.optimizer.step()

                    batch_X, batch_entity, batch_multiword, batch_sameas, batch_relation = [], [], [], [], []

                # Display
                if (batch_idx+1) % self.batch_status == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTotal Loss: {:.6f}'.format(
                    epoch, batch_idx+1, len(self.train_X),
                    100. * batch_idx / len(self.train_X), float(loss), round(sum(losses) / len(losses), 5)))
            
            # TO DO EVALUATION

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