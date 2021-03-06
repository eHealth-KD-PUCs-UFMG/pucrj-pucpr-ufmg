
from transformers import AutoTokenizer  # Or BertTokenizer
from transformers import AutoModel  # or BertModel, for BERT without pretraining heads
from transformers import MT5EncoderModel, T5Tokenizer
import torch
import torch.nn as nn
import utils

class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()

        self.linear1 = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.linear2 = nn.Linear(input_dim, output_dim)
        self.softmax = nn.LogSoftmax(2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = x * self.tanh(self.softplus(x))
        x = self.linear2(x)
        return x, self.softmax(x)
        

class JointEntityRelation(nn.Module):
    def __init__(self, pretrained_model_path='dccuchile/bert-base-spanish-wwm-cased',
                 hdim=768, edim=len(utils.ENTITIES), rdim=len(utils.RELATIONS),
                 device='cuda', max_length=128):
        super(JointEntityRelation, self).__init__()
        self.hdim = hdim
        self.edim = edim
        self.rdim = rdim
        self.device = device
        self.max_length = max_length
        self.pretrained_model_path = pretrained_model_path
        
        # BETO
        if 'mt5' in pretrained_model_path:
            # google/mt5-base
            self.beto = MT5EncoderModel.from_pretrained(pretrained_model_path)
            self.tokenizer = T5Tokenizer.from_pretrained(pretrained_model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, do_lower_case=False)
            self.beto = AutoModel.from_pretrained(pretrained_model_path)

        self.linear_layer = nn.Linear(2 * (hdim + edim), hdim)
        self.tanh = nn.Tanh()

        # linear projections
        self.entity_classifier = Classifier(hdim, edim)

        self.related_classifier = Classifier(hdim, rdim)

    def forward(self, texts):
        # part 1
        tokens = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
        embeddings = self.beto(**tokens)['last_hidden_state']

        # part 2
        logits, entity = self.entity_classifier(embeddings)

        # part 23
        embeddings_entity = torch.cat([embeddings, logits], 2)

        # part 3
        batch, seq_len, dim = embeddings_entity.size()
        embent_embent = torch.zeros(batch, seq_len**2, 2*dim).to(self.device)

        for i in range(seq_len):
            m1 = torch.cat(seq_len * [embeddings_entity[:, i, :].unsqueeze(1)], 1)
            m2 = torch.cat([m1, embeddings_entity], 2)
            embent_embent[:, seq_len*i:seq_len*i+seq_len, :] = m2

        # part 4
        ssd = self.tanh(self.linear_layer(embent_embent))

        # part 5
        logits, related = self.related_classifier(ssd)

        return entity, related

class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, entity_loss, relation_loss):

        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0*entity_loss + self.log_vars[0]

        precision1 = torch.exp(-self.log_vars[1])
        loss1 = precision1*relation_loss + self.log_vars[1]
        
        return loss0+loss1