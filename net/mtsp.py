import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
# DEVICE = torch.device('cpu')


#网络超参数
embeddingSize = 256 # 节点的嵌入维度
nodeSize = 21  # 节点总数
agentSize = 3  # 车辆数
# batch = 128  # batch size
batch = 512 # batch size
M:int = 8  # 多头注意力中的头数
dk:int = embeddingSize / M  # 多头注意力中每一头的维度
isTrain = True  # 是否训练
C = 10  # 做softmax得到选取每个点概率前，clip the result所使用的参数

class ActNet(nn.Module):
    def __init__(self, embedding_size:int=embeddingSize, node_size:int=nodeSize,agent_size:int = agentSize, batch:int=batch, m:int=M, dk:int=dk, feature_dim:int=3):
        super(ActNet, self).__init__()
        self.embedding_size = embedding_size
        self.node_size = node_size
        self.agent_size = agent_size
        self.batch = batch
        self.M = M
        self.dk = int(dk)
        self.feature_dim = feature_dim

        # 嵌入层
        self.embedding = nn.Linear(self.feature_dim, embedding_size)

        # encoder
        self.wq1 = nn.Linear(embedding_size, embedding_size)
        self.wk1 = nn.Linear(embedding_size, embedding_size)
        self.wv1 = nn.Linear(embedding_size, embedding_size)
        self.w1 = nn.Linear(embedding_size, embedding_size)
        self.ffc11 = nn.Linear(embedding_size, embedding_size)
        self.ffc12 = nn.Linear(embedding_size, embedding_size)
        self.bn11 = nn.BatchNorm1d(embedding_size)
        self.bn12 = nn.BatchNorm1d(embedding_size)

        self.wq2 = nn.Linear(embedding_size, embedding_size)
        self.wk2 = nn.Linear(embedding_size, embedding_size)
        self.wv2 = nn.Linear(embedding_size, embedding_size)
        self.w2 = nn.Linear(embedding_size, embedding_size)
        self.ffc21 = nn.Linear(embedding_size, embedding_size)
        self.ffc22 = nn.Linear(embedding_size, embedding_size)
        self.bn21 = nn.BatchNorm1d(embedding_size)
        self.bn22 = nn.BatchNorm1d(embedding_size)

        self.wq3 = nn.Linear(embedding_size, embedding_size)
        self.wk3 = nn.Linear(embedding_size, embedding_size)
        self.wv3 = nn.Linear(embedding_size, embedding_size)
        self.w3 = nn.Linear(embedding_size, embedding_size)
        self.ffc31 = nn.Linear(embedding_size, embedding_size)
        self.ffc32 = nn.Linear(embedding_size, embedding_size)
        self.bn31 = nn.BatchNorm1d(embedding_size)
        self.bn32 = nn.BatchNorm1d(embedding_size)

        # decoder
        self.wq4 = nn.Linear(embedding_size*2, embedding_size)
        self.wk4 = nn.Linear(embedding_size, embedding_size)
        self.wv4 = nn.Linear(embedding_size, embedding_size)
        self.w4 = nn.Linear(embedding_size, embedding_size)
        # 输出层
        self.wq5 = nn.Linear(embedding_size, embedding_size)
        self.wk5 = nn.Linear(embedding_size, embedding_size)


    def forward(self, x_, is_train):
        s1 = torch.unsqueeze(x_, dim=1)
        s1 = s1.expand(self.batch, self.node_size, self.node_size, self.feature_dim)
        s2 = torch.unsqueeze(x_, dim=2)
        s2 = s2.expand(self.batch, self.node_size, self.node_size, self.feature_dim)
        ss = s1 - s2
        ss = ss[:,:,:,0:2]
        dis = torch.norm(ss, 2, dim=3, keepdim=True)  # dis表示任意两点间的距离 (batch, node_size, node_size, 1)
        dis[:,:,0:self.agent_size,:] = 0
        # 嵌入层
        x = self.embedding(x_)#(batch, node_size, embedding_size)
        # encoder
        # 第一层
        x = self.attention(x, self.wq1, self.wk1, self.wv1, self.w1, add_residual=True)
        x = self.bn11(x.permute(0, 2, 1)).permute(0, 2, 1)#BatchNorm1d对二维中的最后一维，或三维中的中间一维进行归一化
        x1 = self.ffc11(x)
        x1 = F.relu(x1)
        x1 = self.ffc12(x1)
        x = x1 + x
        x = self.bn12(x.permute(0, 2, 1)).permute(0, 2, 1)
        # 第二层
        x = self.attention(x, self.wq2, self.wk2, self.wv2, self.w2, add_residual=True)
        x = self.bn21(x.permute(0, 2, 1)).permute(0, 2, 1)
        x1 = self.ffc21(x)
        x1 = F.relu(x1)
        x1 = self.ffc22(x1)
        x = x1 + x
        x = self.bn22(x.permute(0, 2, 1)).permute(0, 2, 1)
        # 第三层
        x = self.attention(x, self.wq3, self.wk3, self.wv3, self.w3, add_residual=True)
        x = self.bn31(x.permute(0, 2, 1)).permute(0, 2, 1)
        x1 = self.ffc31(x)
        x1 = F.relu(x1)
        x1 = self.ffc32(x1)
        x = x1 + x
        x = self.bn32(x.permute(0, 2, 1)).permute(0, 2, 1)

        #(batch, node_size, embedding_size) -> (batch, embedding_size)
        ave = torch.mean(x, dim=1)

        # decoder

        idx = torch.zeros(self.batch,dtype=torch.long).to(DEVICE)  # 当前车辆所在的点
        idx_last = torch.zeros(self.batch,dtype=torch.long).to(DEVICE)  # 上一个车辆所在的点
        mask = torch.zeros(self.batch, self.node_size,dtype=torch.bool).to(DEVICE)
        pro = torch.FloatTensor(self.batch, self.node_size-1).to(DEVICE)  # 每个点被选取时的选取概率,将其连乘可得到选取整个路径的概率
        distance = torch.zeros(self.batch).to(DEVICE)  # 总距离
        seq = torch.zeros(self.batch, self.node_size-1).to(DEVICE)  # 选择的路径序列
        for i in range(self.node_size-1):
            #退出条件
            # if i == self.node_size - 1:
            #     break
            mask_temp = torch.zeros(self.batch, self.node_size,dtype=torch.bool).to(DEVICE)
            mask_temp[torch.arange(self.batch), idx_last] = 1 
            mask = mask | mask_temp
            # for j in range(self.batch):
            #     mask[j, idx_last[j]] = 1
            mask_ = mask.unsqueeze(1)#(batch, 1, node_size)
            mask_ = mask_.expand(self.batch, self.M, self.node_size)
            mask_ = mask_.unsqueeze(2)#(batch, M, 1, node_size)
                
            # now = x[:, idx, :]
            now = x[torch.arange(self.batch), idx, :]
            graph_info = torch.cat([ave, now], dim=1)#(batch, 2*embedding_size)
            q = self.wq4(graph_info)
            k = self.wk4(x)
            v = self.wv4(x)
            q = q.contiguous().view(self.batch, 1, self.M, self.dk)
            k = k.contiguous().view(self.batch, self.node_size, self.M, self.dk)
            v = v.contiguous().view(self.batch, self.node_size, self.M, self.dk)
            q = q.permute(0, 2, 1, 3)#(batch, M, 1, dk)
            k = k.permute(0, 2, 3, 1)#(batch, M, dk, node_size)
            v = v.permute(0, 2, 1, 3)#(batch, M, node_size, dk)
            qk = torch.matmul(q, k) / (self.dk ** 0.5)#(batch, M, 1, node_size)
            #mask
            qk.masked_fill_(mask_, -float('inf'))
            qk = F.softmax(qk, dim=-1)#(batch, M, 1, node_size)
            z = torch.matmul(qk, v)#(batch, M, 1, dk)
            z = z.permute(0, 2, 1, 3)#(batch, 1, M, dk)
            z = z.contiguous().view(self.batch, 1, self.embedding_size)
            z = self.w4(z)#(batch, 1, embedding_size)

            #输出概率
            q = self.wq5(z)#(batch, 1, embedding_size)
            k = self.wk5(x)#(batch, node_size, embedding_size)
            k = k.permute(0, 2, 1)#(batch, embedding_size, node_size)
            qk = torch.matmul(q, k) / (self.dk ** 0.5)#(batch, 1, node_size)
            qk = qk.sum(dim=1)#(batch, node_size)
            qk = torch.tanh(qk)*C#TODO: c待查
            qk.masked_fill_(mask, -float('inf'))
            p = F.softmax(qk, dim=-1)#(batch, node_size)

            if is_train:
                idx = torch.multinomial(p, 1)[:, 0]#(batch, 1) ->[:,0]->(batch)
            else:
                idx = torch.argmax(p, dim=1)#(batch, node_size) ->(batch)

            pro[:,i] = p[torch.arange(self.batch), idx]
            distance = distance + dis[torch.arange(self.batch), idx_last, idx].squeeze()

            idx_last = idx
            seq[:, i] = idx.squeeze()

        if is_train==False:
            seq = seq.detach()
            pro = pro.detach()
            distance = distance.detach()
        
        return seq, pro, distance




    def attention(self, x, wq, wk, wv, w, add_residual):
        q = wq(x)
        k = wk(x)
        v = wv(x)
        q = q.contiguous().view(self.batch, self.node_size, self.M, self.dk)
        k = k.contiguous().view(self.batch, self.node_size, self.M, self.dk)
        v = v.contiguous().view(self.batch, self.node_size, self.M, self.dk)
        q = q.permute(0, 2, 1, 3)#(batch, M, node_size, dk)
        k = k.permute(0, 2, 3, 1)#(batch, M, dk, node_size)
        v = v.permute(0, 2, 1, 3)#(batch, M, node_size, dk)
        qk = torch.matmul(q, k) / (self.dk ** 0.5)#(batch, M, node_size, node_size)
        qk = F.softmax(qk, dim=-1)
        z = torch.matmul(qk, v)#(batch, M, node_size, dk)
        z = z.permute(0, 2, 1, 3)#(batch, node_size, M, dk)
        z = z.contiguous().view(self.batch, self.node_size, self.embedding_size)
        z = w(z)
        if add_residual:
            z = z + x
        return z


if __name__ == '__main__':
    act_net = ActNet()
    act_net = act_net.to(DEVICE)
    x = torch.rand([batch, nodeSize, 3])
    x = x.to(DEVICE)
    seq, pro, distance = act_net(x, isTrain)
    print(seq)
    print(pro)
    print(distance)
    print(seq.size())
    print(pro.size())
    print(distance.size())
    print(seq.dtype)
    print(pro.dtype)
    print(distance.dtype)