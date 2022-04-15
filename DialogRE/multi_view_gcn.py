import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
class MultiRelationalGCN(nn.Module):
    """ A GCN/Contextualized GCN module operated on dependency graphs. """

    def __init__(self, args, config):
        super(MultiRelationalGCN, self).__init__()
        self.in_dim = config.hidden_size
        self.mem_dim = args.graph_hidden_size
        self.args = args

        self.in_drop = nn.Dropout(args.input_dropout)
        self.gcn_drop = nn.Dropout(args.gcn_dropout)
        self.layers_num = args.graph_layers

        self.linear_W = nn.Linear(self.in_dim, self.mem_dim)

        # 多头注意力
        self.heads = args.heads
        self.sublayer_first = args.sublayer_first
        self.sublayer_second = args.sublayer_second
        self.attn = MultiHeadDualAttention(self.heads, self.mem_dim)

        # gcn layer
        self.layers = nn.ModuleList()

        # gcn layer
        for i in range(self.layers_num):
            if i == 0:
                self.layers.append(MultiGraphConvLayer(args, self.mem_dim, self.sublayer_first, self.heads))
                self.layers.append(MultiGraphConvLayer(args, self.mem_dim, self.sublayer_second, self.heads))
            else:
                self.layers.append(MultiGraphConvLayer(args, self.mem_dim, self.sublayer_first, self.heads))
                self.layers.append(MultiGraphConvLayer(args, self.mem_dim, self.sublayer_second, self.heads))

        self.agg_nodes_num = int(len(self.layers) * self.mem_dim)
        self.aggregate_W = nn.Linear(self.agg_nodes_num, self.mem_dim)

        self.GraphGenerate = GraphGenerate(args, config)

        self.aggregate_W = self.aggregate_W.cuda()
        self.linear_W = self.linear_W.cuda()

    def layer_attention(self, layer_list, mask):
        h_out_list = []
        for l in range(len(layer_list)):
            h_out = pool(layer_list[l], mask, type="max")
            # h_out = self.pool_attention(layer_list[l], mask)
            h_out_list.append(h_out.unsqueeze(1))

        agg_out = torch.cat(h_out_list, dim=1)
        score = self.attention_W(agg_out)
        score = torch.softmax(score, dim=1).squeeze(2)
        out = score.unsqueeze(1).bmm(agg_out).squeeze(1)

        return out

    def forward(self, adj, inputs, input_ids):
        src_mask = (input_ids == 0).unsqueeze(-2)
        src_mask = src_mask[:, :, :adj.size(2)]

        inputs = self.in_drop(inputs)

        gcn_inputs = self.linear_W(inputs)
        # gcn layer

        gcn_outputs = gcn_inputs
        layer_list = []

        mask_out = src_mask.reshape([src_mask.size(0), src_mask.size(2), src_mask.size(1)])
        _, attn_adj_list = self.GraphGenerate(gcn_inputs, adj)

        for i in range(len(self.layers)):
            if i < 2:
                attn_adj_list, gcn_outputs, src_mask = self.layers[i](attn_adj_list, gcn_outputs, src_mask)
                layer_list.append(gcn_outputs)
                if i == 0:
                    src_mask_input = src_mask
            else:
                attn_tensor = self.attn(gcn_outputs, gcn_outputs, gcn_outputs, src_mask)
                attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
                attn_adj_list, gcn_outputs, src_mask = self.layers[i](attn_adj_list, gcn_outputs, src_mask)
                layer_list.append(gcn_outputs)

        aggregate_out = torch.cat(layer_list, dim=2)
        aggregate_out = self.aggregate_W(aggregate_out)

        out = pool(aggregate_out, mask_out, 'max')


        return out, mask_out, layer_list, src_mask_input

    @staticmethod
    def entity_logsumexp(hidden_output, e_mask):
        e_mask = e_mask.unsqueeze(2)
        hidden_output = hidden_output.masked_fill(e_mask == 0, 0)
        vector = torch.logsumexp(hidden_output, dim=1)
        return vector

    @staticmethod
    def att_mean(hidden_output, e_mask):
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)
        e_mask = e_mask.unsqueeze(2)
        hidden_output = hidden_output.masked_fill(e_mask == 0, 0)
        sum_vector = torch.sum(hidden_output, dim=1)
        avg_vector = sum_vector / length_tensor
        return avg_vector

    @staticmethod
    def entity_max(hidden_output, e_mask):
        e_mask = e_mask.unsqueeze(2)
        hidden_output = hidden_output.masked_fill(e_mask == 0, -1e4)
        vector = torch.max(hidden_output, dim=1)[0]
        return vector

    @staticmethod
    def entity_mean(hidden_output, e_mask):
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)
        e_mask = e_mask.unsqueeze(2)
        hidden_output = hidden_output.masked_fill(e_mask == 0, 0)
        sum_vector = torch.sum(hidden_output, dim=1)
        avg_vector = sum_vector / length_tensor
        return avg_vector

    @staticmethod
    def entity_logsumexp(hidden_output, e_mask):
        e_mask = e_mask.unsqueeze(2)
        hidden_output = hidden_output.masked_fill(e_mask == 0, -1e4)
        vector = torch.logsumexp(hidden_output, dim=1)
        return vector

class PoolAttention(nn.Module):
    def __init__(self, dimensions):
        super(PoolAttention, self).__init__()

        self.linear_out = nn.Linear(dimensions, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.linear_out = self.linear_out.cuda()

    def forward(self, context, mask=None):
        attention_scores = self.linear_out(context)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask, -1e4)
        attention_weights = self.softmax(attention_scores).squeeze(2)
        out = attention_weights.unsqueeze(1).bmm(context).squeeze(1)

        return out
class GraphGenerate(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.num_heads = args.heads
        self.hid_dim = args.graph_hidden_size
        self.R = self.hid_dim // self.num_heads

        self.subj_transform = nn.Linear(args.graph_hidden_size * 2, args.graph_hidden_size)
        self.obj_transform = nn.Linear(args.graph_hidden_size * 2, args.graph_hidden_size)
        self.transform = nn.Linear(args.graph_hidden_size, self.num_heads)
        self.dropout = nn.Dropout(args.input_dropout)
        self.pool_X = nn.MaxPool1d(args.max_offset * 2 + 1, stride=1, padding=args.max_offset)
        self.transform = self.transform.cuda()

    def forward(self, x, adj, mask=None):
        B, T, C = x.size()

        x_pooled = self.pool_X(x)
        x_new = torch.cat([x_pooled, x], dim=2)

        subj = self.subj_transform(x_new)
        subj = self.dropout(subj)
        subj = subj.unsqueeze(1).repeat(1, T, 1, 1)

        obj = self.obj_transform(x_new)
        obj = self.dropout(obj)
        obj = obj.unsqueeze(2).repeat(1, 1, T, 1)

        ### 绝对差值构图
        abs_multi = torch.abs(subj - obj)
        rep_multi = self.transform(abs_multi)
        attn_adj = torch.softmax(torch.relu(rep_multi), dim=2)
        attn_adj_list = [attn_adj.squeeze(3) for attn_adj in torch.split(attn_adj, 1, dim=3)]

        if mask is not None:
            mask_len = mask.size(2)
            mask_matrix_1 = mask.repeat(1, mask_len, 1)
            mask_matrix_2 = mask.squeeze(1).unsqueeze(2).repeat(1, 1, mask_len)
            mask_matrix = mask_matrix_1.mul(mask_matrix_2)
            for i in range(len(attn_adj_list)):
                 attn_adj_list[i] = attn_adj_list[i].masked_fill(mask_matrix, 0)

        return x_new, attn_adj_list

def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -1e4)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)

class GraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, args, mem_dim, layers):
        super(GraphConvLayer, self).__init__()
        self.opt = args
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.gcn_drop = nn.Dropout(args.gcn_dropout)

        # linear transformation
        self.linear_output = nn.Linear(self.mem_dim, self.mem_dim)

        # gcn block
        self.weight_list = nn.ModuleList()
        for i in range(self.layers):
            self.weight_list.append(nn.Linear((self.mem_dim + self.head_dim * i), self.head_dim))

        self.weight_list = self.weight_list.cuda()
        self.linear_output = self.linear_output.cuda()

    def forward(self, adj, gcn_inputs, src_mask):
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1

        outputs = gcn_inputs
        cache_list = [outputs]
        output_list = []
        for l in range(self.layers):
            Ax = adj.bmm(outputs)
            AxW = self.weight_list[l](Ax)
            AxW = AxW + self.weight_list[l](outputs)  # self loop
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            cache_list.append(gAxW)
            outputs = torch.cat(cache_list, dim=2)
            output_list.append(self.gcn_drop(gAxW))

        gcn_outputs = torch.cat(output_list, dim=2)

        gcn_outputs = gcn_outputs + gcn_inputs
        out = self.linear_output(gcn_outputs)

        return adj, out, src_mask

class MultiGraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, args, mem_dim, layers, heads):
        super(MultiGraphConvLayer, self).__init__()
        self.args = args
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.heads = heads
        self.gcn_drop = nn.Dropout(args.gcn_dropout)
        self.pool = Dynamic_Pool_Multi(args, ratio=args.pooling_ratio, heads=self.heads)
        # layer
        self.Linear = nn.Linear(self.mem_dim * self.heads, self.mem_dim)
        self.weight_list = nn.ModuleList()

        for i in range(self.heads):
            for j in range(self.layers):
                self.weight_list.append(nn.Linear(self.mem_dim + self.head_dim * j, self.head_dim))

        self.weight_list = self.weight_list.cuda()
        self.Linear = self.Linear.cuda()

    def forward(self, adj_list, gcn_inputs, src_mask):

        multi_head_list = []
        for i in range(self.heads):
            adj = adj_list[i]
            denom = adj.sum(2).unsqueeze(2) + 1
            outputs = gcn_inputs
            cache_list = [outputs]
            output_list = []
            for l in range(self.layers):
                index = i * self.layers + l
                Ax = adj.bmm(outputs)
                AxW = self.weight_list[index](Ax)
                AxW = AxW + self.weight_list[index](outputs)  # self loop
                AxW = AxW / denom
                gAxW = F.relu(AxW)
                cache_list.append(gAxW)
                outputs = torch.cat(cache_list, dim=2)
                output_list.append(self.gcn_drop(gAxW))

            gcn_outputs = torch.cat(output_list, dim=2)
            gcn_outputs = gcn_outputs + gcn_inputs

            multi_head_list.append(gcn_outputs)

        multi_head_list, src_mask = self.pool(multi_head_list, src_mask)
        final_output = torch.cat(multi_head_list, dim=2)
        out = self.Linear(final_output)

        return adj_list, out, src_mask

def Top_K(score, ratio):
    # batch_size = score.size(0)
    node_sum = score.size(1)
    score = score.view(-1,node_sum)
    K = int(ratio*node_sum)
    '''if K <=1:
        K = K + 1'''
    Top_K_values, Top_K_indices = score.topk(K, largest=False, sorted=False)

    return Top_K_values, Top_K_indices


class Dynamic_Pool_Multi(torch.nn.Module):
    def __init__(self, args, ratio=0.5, heads=3):
        super(Dynamic_Pool_Multi,self).__init__()

        self.drop_ratio = 1 - ratio
        self.score_layer = nn.Linear(args.graph_hidden_size, 1)

    def forward(self, x_list, src_mask):

        src_mask_list = []

        for x in x_list:
            score = self.score_layer(x)

            _, idx = Top_K(score, self.drop_ratio)

            src_mask_clone = src_mask.clone()
            # 优化流程
            for i in range(src_mask.size(0)):
                src_mask_clone[i][0][idx[i]] = True

            # for i in range(src_mask.size(0)):
            #     for j in range(idx.size(1)):
            #         src_mask_clone[i][0][idx[i][j]] = True

            src_mask_list.append(src_mask_clone)

        src_mask_out = torch.zeros_like(src_mask_list[0]).cuda()
        for src_mask_i in src_mask_list:
            src_mask_out = src_mask_out + src_mask_i

        return x_list, src_mask_out

class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]
        attn = attention(query, key, mask=mask, dropout=self.dropout)

        return attn

class MultiHeadDualAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadDualAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(DualAttention(d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, u_input, mask):
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        query, key = [l(x, u_input).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]
        attn = attention(query, key, mask=mask, dropout=self.dropout)

        return attn

class DualAttention(nn.Module):
    def __init__(self, mem_dim):
        super(DualAttention, self).__init__()
        self.linear = nn.Linear(mem_dim, mem_dim)
        self.U_linear = nn.Linear(mem_dim, mem_dim)
        self.P = nn.Linear(mem_dim, 1)

        self.U_linear = self.U_linear.cuda()
        self.P = self.P.cuda()
        self.linear = self.linear.cuda()

    def forward(self, input, u):
        input = self.linear(input)
        u_input = self.U_linear(u)
        p_u_input = self.P(u_input)
        p_input = self.P(input)
        para = torch.sigmoid(p_input + p_u_input)
        output = (1 - para) * input + para * u_input
        return output

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e4)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn

class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0):
        super(LinearLayer, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)