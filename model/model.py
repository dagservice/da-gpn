import torch
import torch.nn as nn
from transformers import RobertaModel
from torch.cuda.amp import autocast
import numpy as np

from model.multi_view_gcn import MultiRelationalGCN, LinearLayer


class REModel(nn.Module):
    def __init__(self, args, config, num_class):
        super().__init__()
        self.args = args
        self.config = config
        self.encoder = RobertaModel.from_pretrained(args.model_name_or_path, config=self.config)

        self.gcn = MultiRelationalGCN(args, self.config)

        classifier_dim = config.hidden_size * 3 + args.graph_hidden_size * 3

        self.classifier = LinearLayer(classifier_dim, num_class, config.hidden_dropout_prob)

        self.loss_fnt = nn.CrossEntropyLoss()
        self.classifier = self.classifier.cuda()

    @autocast()
    def forward(self, input_ids=None, attention_mask=None, labels=None, s_mask=None, o_mask=None):
        seq_length = input_ids.size(1)
        attention_mask_ = attention_mask.view(-1, seq_length)
        l = (attention_mask_.data.cpu().numpy() != 0).astype(np.int64).sum(1)
        real_length = max(l)

        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
        )
        word_embedding = outputs[0]
        pooled_output = outputs[1]

        ss_emb = self.entity_mean(word_embedding, s_mask)
        os_emb = self.entity_mean(word_embedding, o_mask)

        word_embedding = word_embedding[:, :real_length]

        adj = torch.ones(input_ids.size(0), real_length, real_length).cuda()

        h_out, pool_mask, layer_list, subj_gcn_entity, obj_gcn_entity = self.gcn(adj, word_embedding, input_ids, s_mask, o_mask)

        outputs = torch.cat([ss_emb, subj_gcn_entity, os_emb, obj_gcn_entity, pooled_output, h_out], dim=1)

        logits = self.classifier(outputs)

        outputs = (logits,)
        if labels is not None:
            loss = self.loss_fnt(logits.float(), labels)
            outputs = (loss,) + outputs
        return outputs

    @staticmethod
    def entity_mean(hidden_output, e_mask):
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)
        e_mask = e_mask.unsqueeze(2)
        hidden_output = hidden_output.masked_fill(e_mask == 0, 0)
        sum_vector = torch.sum(hidden_output, dim=1)
        avg_vector = sum_vector / length_tensor
        return avg_vector

