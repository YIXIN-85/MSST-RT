import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as NP



# # NTU
# smask = torch.tensor([[True, True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True],
#                      [True, True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True]])
# seq_len = torch.tensor([4, 3, 3, 2, 3, 3, 3, 4, 3, 3, 3, 4, 3, 3, 3, 2, 3, 3, 3, 2, 5, 2, 2, 2, 2])
#
# graph = {'1': [1, 2, 13, 17], '2': [1, 2, 21], '3': [3, 4, 21],'4': [3, 4],
#          '5': [5, 6, 21], '6': [5, 6, 7], '7': [6, 7, 8], '8': [7, 8, 22, 23],
#          '9': [9, 10, 21], '10': [9, 10, 11], '11': [10, 11, 12], '12': [11, 12, 24, 25],
#          '13': [1, 13, 14], '14': [13, 14, 15], '15': [14, 15, 16], '16': [15, 16],
#          '17': [1, 17, 18], '18': [17, 18, 19], '19': [18, 19, 20], '20': [19, 20],
#          '21': [2, 3, 5, 9, 21], '22': [8, 22], '23': [8, 23], '24': [12, 24], '25': [12, 25]}


# UAV
smask = torch.tensor([[True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
                     [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]])
seq_len = torch.tensor([4, 2, 2, 1, 1, 3, 3, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1])

graph = {'1': [1, 2, 3, 6, 7], '2': [1, 2, 4], '3': [1, 3, 5],'4': [2, 4],
         '5': [3, 5], '6': [1, 6, 8, 12], '7': [1, 7, 9, 13], '8': [6, 8, 10],
         '9': [7, 9, 13], '10': [8, 10], '11': [9, 11], '12': [9, 12, 14],
         '13': [11, 13, 15], '14': [12, 14, 16], '15': [13, 15, 17], '16': [14, 16],
         '17': [15, 17]}


class SRT(nn.Module):
    ''' Compose with four layers '''
    def __init__(self, d, hidden_size, num_layers, num_head, dk, dv, d_in, dropout):
        super(SRT, self).__init__()
        self.iters = num_layers

        self.norm = nn.ModuleList([nn.LayerNorm(hidden_size, eps=1e-6) for _ in range(self.iters)])
        self.d_in = d_in
        self.emb_drop = nn.Dropout(dropout)
        self.SJU_ALL = nn.ModuleList(
            [sju(d, hidden_size, nhead=num_head, dk=dk, dv=dv, d_in=d_in, dropout=0.0)
             for _ in range(self.iters)])
        self.SRU_ALL = nn.ModuleList(
            [sru(hidden_size, nhead=num_head, dk=dk, dv=dv, d_in=d_in, dropout=0.0)
             for _ in range(self.iters)])


    def forward(self, data): # B C V T

        def norm_func(f, x):
            return f(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        B, C, V, T = data.size()

        # B*T C V 1
        embs = data.permute(0, 3, 1, 2).contiguous().view(B * T, C, V, 1)
        embs = norm_func(self.emb_drop, embs)

        nodes = embs
        relay = embs.mean(2, keepdim=True)
        r_embs = embs.view(B * T, C, 1, V)
        nodes_cat = list()
        for i in range(self.iters):
            ax = torch.cat([r_embs, relay.expand(B * T, C, 1, V)], 2)
            nodes = F.leaky_relu(self.SJU_ALL[i](norm_func(self.norm[i], nodes), ax=ax))
            relay = F.leaky_relu(self.SRU_ALL[i](relay, torch.cat([relay, nodes], 2)))
            nodes_cat.append(nodes)

        nodes = nodes.view(B, T, C, V).permute(0, 2, 3, 1)

        return nodes


class sju(nn.Module):
    def __init__(self, d, nhid, nhead, dk, dv, d_in, dropout=0.1):
        super(sju, self).__init__()
        self.nhid = nhid
        self.nhead = nhead
        self.dk = dk
        self.dv = dv
        # Multi-head Attention
        self.WQ = nn.Conv2d(nhid, nhead * dk, 1)
        self.WK = nn.Conv2d(nhid, nhead * dk, 1)
        self.WV = nn.Conv2d(nhid, nhead * dv, 1)

        self.drop = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(nhid, eps=1e-6)
        self.batch_norm = nn.BatchNorm1d(nhid)
        self.pos_ffn = PositionFeedForward(nhead * dk, d_in, dropout=dropout)

        self.pad1 = nn.ZeroPad2d(padding=(0, 0, 0, 1))
        self.pad2 = nn.ZeroPad2d(padding=(0, 0, 0, 2))
        self.pad3 = nn.ZeroPad2d(padding=(0, 0, 0, 3))

        self.mask = self.seq_len_to_mask(d, seq_len, max_len=5)

    def forward(self, x, ax=None):
        # x: B, H, L, 1,  ax : B, H, 2, L append features
        nhid, nhead, dk, dv = self.nhid, self.nhead, self.dk, self.dv
        mask = self.mask.to(x)
        B, C, L, _ = x.shape
        q, k, v = self.WQ(x), self.WK(x), self.WV(x)
        if ax is not None:
            aL = ax.shape[2]
            ak = self.WK(ax).view(B, nhead, dk, aL, L)
            av = self.WV(ax).view(B, nhead, dv, aL, L)
        q = q.view(B, nhead, dk, 1, L)
        k = self.unfold(k.view(B, nhead * dk, L, 1)).view(B, nhead, dk, 5, L)
        v = self.unfold(v.view(B, nhead * dv, L, 1)).view(B, nhead, dv, 5, L)
        if ax is not None:
            k = torch.cat([k, ak], 3)
            v = torch.cat([v, av], 3)
        pre_a = (q * k).sum(2, keepdim=True) / NP.sqrt(dk)

        if mask is not None:
            pre_a = pre_a.masked_fill(mask[None, None, None, :, :].permute(0, 1, 2, 4, 3)==0, -float('inf'))
        alphas = self.drop(F.softmax(pre_a, 3))
        att = (alphas * v).sum(3).view(B, nhead * dv, L, 1)

        ret = x + att
        ret = self.batch_norm(ret.view(B, -1, L)).view(B, -1, L, 1)
        ret = self.pos_ffn(ret)

        return ret

    def seq_len_to_mask(self, d, seq_len, max_len=None):
        assert seq_len.dim() == 1, f"seq_len can only have one dimension, got {seq_len.dim() == 1}."
        batch_size = seq_len.size(0)
        max_len = int(max_len) if max_len else seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
        mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
        sseq = torch.full((d, 1), 2)
        sbr = torch.arange(2).expand(batch_size,-1).to(seq_len)
        smask = sbr.lt(sseq.long())
        mask = torch.cat((mask, smask), dim=1)

        return mask


    def unfold(self,x):     # B, nhead * dk, L, 1
        B, C, L, _ = x.shape
        s = 0
        for i in graph:
            if len(graph[i]) == 2:
                k = torch.cat([x[:, :, graph[i][0]-1, :][:, :, None, :], x[:, :, graph[i][1]-1, :][:, :, None, :]], dim=2)
                k = self.pad3(k)
            elif len(graph[i]) == 3:
                k = torch.cat([x[:, :, graph[i][0]-1, :][:, :, None, :], x[:, :, graph[i][1]-1, :][:, :, None, :], x[:, :, graph[i][2]-1, :][:, :, None, :]], dim=2)
                k = self.pad2(k)
            elif len(graph[i]) == 4:
                k = torch.cat([x[:, :, graph[i][0]-1, :][:, :, None, :], x[:, :, graph[i][1]-1, :][:, :, None, :], x[:, :, graph[i][2]-1, :][:, :, None, :], x[:, :, graph[i][3]-1, :][:, :, None, :]], dim=2)
                k = self.pad1(k)
            else:
                k = torch.cat([x[:, :, graph[i][0]-1, :][:, :, None, :], x[:, :, graph[i][1]-1, :][:, :, None, :], x[:, :, graph[i][2]-1, :][:, :, None, :], x[:, :, graph[i][3]-1, :][:, :, None, :], x[:, :, graph[i][4]-1, :][:, :, None, :]], dim=2)
            if s == 0:
                y = k
                s += 1
            else:
                y = torch.cat([y, k], dim=3)
        return y

class sru(nn.Module):
    def __init__(self, nhid, nhead, dk, dv, d_in, dropout=0.1):
        super(sru, self).__init__()
        self.WQ = nn.Conv2d(nhid, nhead * dk, 1)
        self.WK = nn.Conv2d(nhid, nhead * dk, 1)
        self.WV = nn.Conv2d(nhid, nhead * dv, 1)

        self.drop = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(nhid)
        self.pos_ffn = PositionFeedForward(nhead * dk, d_in, dropout=dropout)

        self.nhid, self.nhead, self.dk, self.dv, self.unfold_size = nhid, nhead, dk, dv, 5

    def forward(self, x, y):
        # x: B, H, 1, 1  y: B H L(26) 1
        nhid, nhead, dk, dv = self.nhid, self.nhead, self.dk, self.dv
        B, H, L, _ = y.shape

        q, k, v = self.WQ(x), self.WK(y), self.WV(y)

        q = q.view(B, nhead, 1, dk)
        k = k.view(B, nhead, dv, L)
        v = v.view(B, nhead, dv, L).permute(0, 1, 3, 2)
        pre_a = torch.matmul(q, k) / NP.sqrt(dk)
        alphas = self.drop(F.softmax(pre_a, 3))


        att = torch.matmul(alphas, v).view(B, -1, 1, 1)

        ret = x.view(B, nhid, 1, 1) + att
        ret = self.batch_norm(ret.view(B, nhid)).view(B, nhid, 1, 1)
        ret = self.pos_ffn(ret)


        return ret

class PositionFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.batch_norm = nn.BatchNorm1d(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = x.permute(0, 2, 3, 1)
        B, L, _, C = x.size()
        residual = x

        x = self.w_2(self.dropout(F.relu(self.w_1(x))))
        x += residual
        del residual

        x = self.batch_norm(x.permute(0, 3, 1, 2).view(B, C, L)).permute(0, 2, 1).view(B, L, 1, C)

        return x.permute(0, 3, 1, 2)

class TRT(nn.Module):
    ''' Compose with four layers '''
    def __init__(self, hidden_size, num_layers, num_head, dk, dv, d_in, dropout):
        super(TRT, self).__init__()
        self.iters = num_layers

        self.norm = nn.ModuleList([nn.LayerNorm(hidden_size, eps=1e-6) for _ in range(self.iters)])
        self.batchnorm = nn.ModuleList([nn.BatchNorm1d(hidden_size, eps=1e-6) for _ in range(self.iters)])
        self.emb_drop = nn.Dropout(dropout)
        self.TJU_ALL = nn.ModuleList(
            [tju(hidden_size, nhead=num_head, dk=dk, dv=dv, d_in=d_in, dropout=0)
             for _ in range(self.iters)])
        self.TRU_ALL = nn.ModuleList(
            [tru(hidden_size, nhead=num_head, dk=dk, dv=dv, d_in=d_in, dropout=0)
             for _ in range(self.iters)])


    def forward(self, data):

        def norm_func(f,x):
            return f(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # B, T, C, V
        B, C, V, T = data.size()
        embs = data.permute(0, 2, 1, 3).contiguous().view(B * V, C, T, 1)
        embs = norm_func(self.emb_drop, embs)
        nodes = embs
        relay = embs.mean(2, keepdim=True)
        r_embs = embs.view(B * V, C, 1, T)
        for i in range(self.iters):
            ax = torch.cat([r_embs, relay.expand(B * V, C, 1, T)], 2)
            nodes = F.leaky_relu(self.TJU_ALL[i](norm_func(self.norm[i], nodes), ax=ax))
            relay= F.leaky_relu(self.TRU_ALL[i](relay, torch.cat([relay, nodes], 2)))
        return nodes.view(B, V, C, T)

class tju(nn.Module):
    def __init__(self, nhid, nhead, dk, dv, d_in, dropout=0.1):
        super(tju, self).__init__()
        # Multi-head Attention

        self.WQ = nn.Conv2d(nhid, nhead * dk, 1)
        self.WK = nn.Conv2d(nhid, nhead * dk, 1)
        self.WV = nn.Conv2d(nhid, nhead * dv, 1)

        self.drop = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(nhid, eps=1e-6)
        self.batch_norm = nn.BatchNorm1d(nhid)
        self.pos_ffn = PositionFeedForward(nhead * dk, d_in, dropout=dropout)

        self.nhid, self.nhead, self.head_dim, self.unfold_size = nhid, nhead, dk, 3

    def forward(self, x, ax=None):
        # x: B, H, L, 1, ax : B, H, X, L append features
        nhid, nhead, head_dim, unfold_size = self.nhid, self.nhead, self.head_dim, self.unfold_size
        B, H, L, _ = x.shape

        q, k, v = self.WQ(x), self.WK(x), self.WV(x)  # x: (B,H,L,1)

        if ax is not None:
            aL = ax.shape[2]
            ak = self.WK(ax).view(B, nhead, head_dim, aL, L)
            av = self.WV(ax).view(B, nhead, head_dim, aL, L)
        q = q.view(B, nhead, head_dim, 1, L)
        k = F.unfold(k.view(B, nhead * head_dim, L, 1), (unfold_size, 1), padding=(unfold_size // 2, 0)) \
            .view(B, nhead, head_dim, unfold_size, L)
        v = F.unfold(v.view(B, nhead * head_dim, L, 1), (unfold_size, 1), padding=(unfold_size // 2, 0)) \
            .view(B, nhead, head_dim, unfold_size, L)
        if ax is not None:
            k = torch.cat([k, ak], 3)
            v = torch.cat([v, av], 3)

        alphas = self.drop(F.softmax((q * k).sum(2, keepdim=True) / NP.sqrt(head_dim), 3))  # B N L 1 U
        att = (alphas * v).sum(3).view(B, nhead * head_dim, L, 1)

        ret = x + att
        ret = self.batch_norm(ret.view(B, -1, L)).view(B, -1, L, 1)
        ret = self.pos_ffn(ret)


        return ret


class tru(nn.Module):
    def __init__(self, nhid, nhead, dk, dv, d_in, dropout=0.1):
        # Multi-head Attention
        super(tru, self).__init__()
        self.WQ = nn.Conv2d(nhid, nhead * dk, 1)
        self.WK = nn.Conv2d(nhid, nhead * dk, 1)
        self.WV = nn.Conv2d(nhid, nhead * dv, 1)

        self.drop = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(nhid, eps=1e-6)
        self.batch_norm = nn.BatchNorm1d(nhid)
        self.pos_ffn = PositionFeedForward(nhead * dk, d_in, dropout=dropout)


        self.nhid, self.nhead, self.dk, self.dv  = nhid, nhead, dk, dv

    def forward(self, x, y, mask=None):
        # x: B, H, 1, 1, 1 y: B H L 1
        nhid, nhead, dk, dv = self.nhid, self.nhead, self.dk, self.dv
        B, H, L, _ = y.shape

        q, k, v = self.WQ(x), self.WK(y), self.WV(y)

        q = q.view(B, nhead, 1, dk)  # B, H, 1, 1 -> B, N, 1, h
        k = k.view(B, nhead, dk, L)  # B, H, L, 1 -> B, N, h, L
        v = v.view(B, nhead, dv, L).permute(0, 1, 3, 2)  # B, H, L, 1 -> B, N, L, h
        pre_a = torch.matmul(q, k) / NP.sqrt(dk) # B, N, 1, L
        alphas = self.drop(F.softmax(pre_a, 3))  # B, N, 1, L
        att = torch.matmul(alphas, v).view(B, -1, 1, 1)

        ret = x.view(B, nhid, 1, 1) + att
        ret = self.batch_norm(ret.view(B, nhid)).view(B, nhid, 1, 1)
        ret = self.pos_ffn(ret)

        return ret




