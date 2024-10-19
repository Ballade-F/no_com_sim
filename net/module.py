import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Self_Attention(nn.Module):
    def __init__(self, embedding_size:int, attention_head:int):
        super(Self_Attention, self).__init__()
        if embedding_size % attention_head != 0 :
            raise ValueError("embedding_size must be divisible by attention_head")
        self.embedding_size = embedding_size
        self.attention_head = attention_head
        self.dk = int(embedding_size / attention_head)
        self.wq = nn.Linear(embedding_size, embedding_size)
        nn.init.kaiming_normal_(self.wq.weight)
        self.wk = nn.Linear(embedding_size, embedding_size)
        nn.init.kaiming_normal_(self.wk.weight)
        self.wv = nn.Linear(embedding_size, embedding_size)
        nn.init.kaiming_normal_(self.wv.weight)
        self.w = nn.Linear(embedding_size, embedding_size)
        nn.init.kaiming_normal_(self.w.weight)
    
    # x: (batch, n, embedding_size)
    def forward(self, x):
        _batch_size = x.shape[0]
        _n = x.shape[1]
        q = self.wq(x) #(batch, n, embedding_size)
        k = self.wk(x) #(batch, n, embedding_size)
        v = self.wv(x) #(batch, n, embedding_size)
        q = q.contiguous().view(_batch_size, _n, self.attention_head, self.dk)
        k = k.contiguous().view(_batch_size, _n, self.attention_head, self.dk)
        v = v.contiguous().view(_batch_size, _n, self.attention_head, self.dk)
        q = q.permute(0, 2, 1, 3) #(batch, attention_head, n, dk)
        k = k.permute(0, 2, 3, 1) #(batch, attention_head, dk, n)
        v = v.permute(0, 2, 1, 3) #(batch, attention_head, n, dk)
        qk = torch.matmul(q, k) / (self.dk ** 0.5) #(batch, attention_head, n, n)
        qk = F.softmax(qk, dim=-1)
        z = torch.matmul(qk, v) #(batch, attention_head, n, dk)
        z = z.permute(0, 2, 1, 3) #(batch, n, attention_head, dk)
        z = z.contiguous().view(_batch_size, _n, self.embedding_size)
        z = self.w(z) #(batch, n, embedding_size)
        z = z + x
        return z

        

# A和AB做注意力
class Self_Cross_Attention(nn.Module):
    def __init__(self, embedding_size:int, attention_head:int):
        super(Self_Cross_Attention, self).__init__()
        if embedding_size % attention_head != 0:
            raise ValueError("embedding_size must be divisible by attention_head")
        self.embedding_size = embedding_size
        self.attention_head = attention_head
        self.dk = int(embedding_size / attention_head)
        self.wq = nn.Linear(embedding_size, embedding_size)
        nn.init.kaiming_normal_(self.wq.weight)
        self.wk_a = nn.Linear(embedding_size, embedding_size)
        nn.init.kaiming_normal_(self.wk_a.weight)
        self.wk_b = nn.Linear(embedding_size, embedding_size)
        nn.init.kaiming_normal_(self.wk_b.weight)
        self.wv_a = nn.Linear(embedding_size, embedding_size)
        nn.init.kaiming_normal_(self.wv_a.weight)
        self.wv_b = nn.Linear(embedding_size, embedding_size)
        nn.init.kaiming_normal_(self.wv_b.weight)
        self.w = nn.Linear(embedding_size, embedding_size)
        nn.init.kaiming_normal_(self.w.weight)

    def forward(self, x_a, x_b):
        _batch_size = x_a.shape[0]
        _n_a = x_a.shape[1]
        _n_b = x_b.shape[1]
        q_a = self.wq(x_a) #(batch, n_a, embedding_size)
        k_a = self.wk_a(x_a) #(batch, n_a, embedding_size)
        v_a = self.wv_a(x_a) #(batch, n_a, embedding_size)
        k_b = self.wk_b(x_b) #(batch, n_b, embedding_size)
        v_b = self.wv_b(x_b) #(batch, n_b, embedding_size)
        q_a = q_a.contiguous().view(_batch_size, _n_a, self.attention_head, self.dk)
        k_a = k_a.contiguous().view(_batch_size, _n_a, self.attention_head, self.dk)
        v_a = v_a.contiguous().view(_batch_size, _n_a, self.attention_head, self.dk)
        k_b = k_b.contiguous().view(_batch_size, _n_b, self.attention_head, self.dk)
        v_b = v_b.contiguous().view(_batch_size, _n_b, self.attention_head, self.dk)
        q_a = q_a.permute(0, 2, 1, 3) #(batch, attention_head, n_a, dk)
        k_a = k_a.permute(0, 2, 3, 1) #(batch, attention_head, dk, n_a)
        v_a = v_a.permute(0, 2, 1, 3) #(batch, attention_head, n_a, dk)
        k_b = k_b.permute(0, 2, 3, 1) #(batch, attention_head, dk, n_b)
        v_b = v_b.permute(0, 2, 1, 3) #(batch, attention_head, n_b, dk)
        k = torch.cat((k_a, k_b), dim=3) #(batch, attention_head, dk, n_a+n_b)
        v = torch.cat((v_a, v_b), dim=2) #(batch, attention_head, n_a+n_b, dk)
        qk_a = torch.matmul(q_a, k) / (self.dk ** 0.5) #(batch, attention_head, n_a, n_a+n_b)
        qk_a = F.softmax(qk_a, dim=-1)
        z_a = torch.matmul(qk_a, v) #(batch, attention_head, n_a, dk)
        z_a = z_a.permute(0, 2, 1, 3) #(batch, n_a, attention_head, dk)
        z_a = z_a.contiguous().view(_batch_size, _n_a, self.embedding_size)
        z_a = self.w(z_a) #(batch, n_a, embedding_size)
        z_a = z_a + x_a
        return z_a
    
    #输入输出[batch, n_points, embedding_size]
class SelfAttentionBlock(nn.Module):
    def __init__(self, embedding_size:int,attention_head:int):
        super(SelfAttentionBlock, self).__init__()
        if embedding_size % attention_head != 0 :
            raise ValueError("embedding_size must be divisible by attention_head")
        
        self.local_attention = Self_Attention(embedding_size, attention_head)
        self.bn1 = nn.BatchNorm1d(embedding_size)
        self.bn2 = nn.BatchNorm1d(embedding_size)
        self.ffc1 = nn.Linear(embedding_size, embedding_size)
        nn.init.kaiming_normal_(self.ffc1.weight)
        self.ffc2 = nn.Linear(embedding_size, embedding_size)
        nn.init.kaiming_normal_(self.ffc2.weight)

    def forward(self, x):
        x = self.local_attention(x)
        x = self.bn1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x1 = self.ffc1(x)
        x1 = F.relu(x1)
        x1 = self.ffc2(x1)
        x = x1 + x
        x = self.bn2(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x