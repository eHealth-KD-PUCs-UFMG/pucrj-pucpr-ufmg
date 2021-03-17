
from transformers import AutoTokenizer  # Or BertTokenizer
from transformers import AutoModel  # or BertModel, for BERT without pretraining heads
from transformers import DistilBertModel, DistilBertConfig
import torch
import torch.nn as nn

class Vicomtech(nn.Module):
    def __init__(self, beto_path='dccuchile/bert-base-spanish-wwm-cased', 
                       hdim=768, edim=5, rdim=13,
                       distilbert_nlayers=2, distilbert_nheads=2):
        super(Vicomtech, self).__init__()
        self.hdim = hdim
        self.edim = edim
        self.rdim = rdim
        
        # BETO
        self.tokenizer = AutoTokenizer.from_pretrained(beto_path, do_lower_case=False)
        self.beto = AutoModel.from_pretrained(beto_path)

        # DistilBERT
        self.distil_layer = nn.Linear(2*(hdim+edim), hdim)
        configuration = DistilBertConfig(vocab_size=len(self.tokenizer.vocab), 
                                        n_layers=distilbert_nlayers, n_heads=distilbert_nheads)
        self.distilbert = DistilBertModel(configuration)

        # linear projections
        self.entity_layer = nn.Linear(hdim, edim)
        self.entity_softmax = nn.Softmax(2)

        self.multiword_layer = nn.Linear(hdim, 2)
        self.multiword_softmax = nn.Softmax(2)

        self.sameas_layer = nn.Linear(hdim, 2)
        self.sameas_softmax = nn.Softmax(2)

        self.related_layer = nn.Linear(hdim, 2)
        self.related_softmax = nn.Softmax(2)

        self.relation_type_layer = nn.Linear(hdim+2, 2)
        self.relation_type_softmax = nn.Softmax(2)

    def forward(self, texts):
        # part 1
        tokens = self.tokenizer(texts, padding=True, return_tensors="pt")
        embeddings = self.beto(**tokens)['last_hidden_state']

        # part 2
        entity = self.entity_softmax(self.entity_layer(embeddings))

        # part 23
        embeddings_entity = torch.cat([embeddings, entity], 2)

        # part 3 (TRY TO IMPROVE VECTORIZATION)
        batch, seq_len, dim = embeddings_entity.size()
        embent_embent = torch.zeros(batch, seq_len**2, 2*dim)

        for i in range(seq_len):
            m1 = torch.cat(seq_len * [embeddings_entity[:, i, :].unsqueeze(1)], 1)
            m2 = torch.cat([m1, embeddings_entity], 2)
            embent_embent[:, seq_len*i:seq_len*i+seq_len, :] = m2

        # thiago addition to adequate dimensions from step 3 to 4
        inp_distilbert = self.distil_layer(embent_embent)
        # part 4
        ssd = self.distilbert(inputs_embeds=inp_distilbert)['last_hidden_state']

        # part 5
        multiword = self.multiword_softmax(self.multiword_layer(ssd))
        # part 6
        sameas = self.sameas_softmax(self.sameas_layer(ssd))
        # part 7
        related = self.related_softmax(self.related_layer(ssd))

        # part 8
        incoming_outgoing = torch.cat([ssd, related], 2)
        # part 9
        related_type = self.relation_type_softmax(self.relation_type_layer(incoming_outgoing))

        return entity, multiword, sameas, related, related_type

