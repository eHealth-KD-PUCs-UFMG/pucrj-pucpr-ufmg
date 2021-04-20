
from transformers import AutoTokenizer  # Or BertTokenizer
from transformers import AutoModel  # or BertModel, for BERT without pretraining heads
from transformers import DistilBertModel, DistilBertConfig
import torch
import torch.nn as nn

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
        

class Vicomtech(nn.Module):
    def __init__(self, pretrained_model_path='dccuchile/bert-base-spanish-wwm-cased',
                 hdim=768, edim=5, rdim=13,
                 distilbert_nlayers=2, distilbert_nheads=2, device='cuda', max_length=128):
        super(Vicomtech, self).__init__()
        self.hdim = hdim
        self.edim = edim
        self.rdim = rdim
        self.device = device
        self.max_length = max_length
        
        # BETO
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, do_lower_case=False)
        self.beto = AutoModel.from_pretrained(pretrained_model_path)

        # DistilBERT
        self.distil_layer = nn.Linear(2*(hdim+edim), hdim)
        self.tanh = nn.Tanh()
        configuration = DistilBertConfig(vocab_size=len(self.tokenizer.vocab), 
                                        n_layers=distilbert_nlayers, n_heads=distilbert_nheads)
        self.distilbert = DistilBertModel(configuration)

        # linear projections
        self.entity_classifier = Classifier(hdim, edim)

        self.multiword_classifier = Classifier(hdim, 2)

        self.sameas_classifier = Classifier(hdim, 2)

        self.related_classifier = Classifier(hdim, 2)

        self.relation_type_classifier = Classifier(hdim+2, rdim)

    def forward(self, texts):
        # part 1
        tokens = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
        embeddings = self.beto(**tokens)['last_hidden_state']

        # part 2
        logits, entity = self.entity_classifier(embeddings)

        # part 23
        embeddings_entity = torch.cat([embeddings, logits], 2)

        # part 3 (TRY TO IMPROVE VECTORIZATION)
        batch, seq_len, dim = embeddings_entity.size()
        embent_embent = torch.zeros(batch, seq_len**2, 2*dim).to(self.device)

        for i in range(seq_len):
            m1 = torch.cat(seq_len * [embeddings_entity[:, i, :].unsqueeze(1)], 1)
            m2 = torch.cat([m1, embeddings_entity], 2)
            embent_embent[:, seq_len*i:seq_len*i+seq_len, :] = m2

        # thiago addition to adequate dimensions from step 3 to 4
        inp_distilbert = self.tanh(self.distil_layer(embent_embent))
        # part 4
        ssd = self.distilbert(inputs_embeds=inp_distilbert)['last_hidden_state']

        # part 5
        _, multiword = self.multiword_classifier(ssd)
        # part 6
        _, sameas = self.sameas_classifier(ssd)
        # part 7
        logits, related = self.related_classifier(ssd)

        # part 8
        incoming_outgoing = torch.cat([ssd, logits], 2)
        # part 9
        _, related_type = self.relation_type_classifier(incoming_outgoing)

        return entity, multiword, sameas, related, related_type

