import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from module import SelfAttentionBlock




# 输入[batch, n_robot, 3]，[batch, n_robot, 3]，[batch, n_obstacle, ob_points, 2]
class AllocationNet(nn.Module):
    def __init__(self, ob_points:int,
                 embedding_size:int, batch_size:int, attention_head:int,
                 rt_dim:int = 3, ob_dim:int = 2, C:float = 10.0,
                 encoder_layer:int = 3, local_embed_layers:int=2, device='cpu'):
        super(AllocationNet, self).__init__()
        if embedding_size % attention_head != 0 :
            raise ValueError("embedding_size must be divisible by attention_head")
        self.robot_n = -1
        self.task_n = -1
        self.ob_n = -1
        self.rt_n = -1
        self.global_points_n = -1
        self.ob_points = ob_points
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.attention_head = attention_head
        self.dk = int(embedding_size / attention_head)
        self.rt_dim = rt_dim
        self.ob_dim = ob_dim
        self.C = C
        self.encoder_layer = encoder_layer
        self.local_embed_layers = local_embed_layers
        self.device = device

        # 嵌入
        self.embedding_rt = nn.Linear(rt_dim, embedding_size)
        nn.init.kaiming_normal_(self.embedding_rt.weight)
        self.embedding_ob = nn.Linear(ob_dim, embedding_size)
        nn.init.kaiming_normal_(self.embedding_ob.weight)

        # encoder
        # local embedding
        self.local_encoder_layers = nn.ModuleList([
            SelfAttentionBlock(embedding_size, attention_head)
            for _ in range(local_embed_layers)
        ])
        # global encoding
        self.encoder_layers = nn.ModuleList([
            SelfAttentionBlock(embedding_size, attention_head)
            for _ in range(encoder_layer)
        ])

        #decoder
        self.dc_wq = nn.Linear(2*embedding_size, embedding_size)
        nn.init.kaiming_normal_(self.dc_wq.weight)
        self.dc_wk = nn.Linear(embedding_size, embedding_size)
        nn.init.kaiming_normal_(self.dc_wk.weight)
        self.dc_wv = nn.Linear(embedding_size, embedding_size)
        nn.init.kaiming_normal_(self.dc_wv.weight)
        self.dc_w = nn.Linear(embedding_size, embedding_size)
        nn.init.kaiming_normal_(self.dc_w.weight)

        #输出层
        self.out_wq = nn.Linear(embedding_size, embedding_size)
        nn.init.kaiming_normal_(self.out_wq.weight)
        self.out_wk = nn.Linear(embedding_size, embedding_size)
        nn.init.kaiming_normal_(self.out_wk.weight)


#x_r: (batch, n_robot, 3), x_t: (batch, n_task, 3), x_ob: (batch, n_obstacle, ob_points, 2), costmap: (batch, n_robot+n_task, n_robot+n_task)
    def forward(self, x_r, x_t, x_ob, costmap, is_train):
        x_rt = torch.cat((x_r, x_t), dim=1)

        #debug
        x_rt_debug = x_rt

        # 嵌入层
        x_rt = self.embedding_rt(x_rt)#(batch, n_robot+n_task, embedding_size)
        x_ob = self.embedding_ob(x_ob)#(batch, n_obstacle, ob_points, embedding_size)

        

        # local embedding
        if x_ob.shape[1] != 0:
            #TODO: 改成vectornet的方法，拼接
            x_ob = x_ob.reshape(self.batch_size*self.ob_n, self.ob_points, self.embedding_size)

            # #debug
            # print(x_ob.shape)

            for layer in self.local_encoder_layers:
                x_ob = layer(x_ob)
            x_ob = x_ob.reshape(self.batch_size, self.ob_n, self.ob_points, self.embedding_size)
        x_ob = torch.mean(x_ob, dim=2)
        # global encoding
        x = torch.cat((x_rt, x_ob), dim=1)

        for layer in self.encoder_layers:
            x = layer(x)
        x_rt = x[:, :self.rt_n, :]
        ave_x_rt = torch.mean(x_rt, dim=1)

        #decoder
        idx = torch.zeros(self.batch_size,dtype=torch.long).to(self.device)  # 当前车辆所在的点
        idx_last = torch.zeros(self.batch_size,dtype=torch.long).to(self.device)  # 上一个车辆所在的点
        mask = torch.zeros(self.batch_size, self.rt_n,dtype=torch.bool).to(self.device)
        pro = torch.FloatTensor(self.batch_size, self.rt_n-1).to(self.device)  # 每个点被选取时的选取概率，将其连乘可得到选取整个路径的概率
        distance = torch.zeros(self.batch_size).to(self.device)  # 总距离
        seq = torch.zeros(self.batch_size, self.rt_n-1).to(self.device)  # 选择的路径序列

        # #debug
        # print(self.batch_size)

        for i in range(self.rt_n-1):
            #mask
            mask_temp = torch.zeros((self.batch_size, self.rt_n),dtype=torch.bool).to(self.device)
            mask_temp[torch.arange(self.batch_size),idx_last] = 1
            mask = mask | mask_temp
            mask_ = mask.unsqueeze(1)#(batch,1,n_rt)
            mask_ = mask_.expand(self.batch_size,self.attention_head,self.rt_n)
            mask_ = mask_.unsqueeze(2)#(batch,attention_head,1,n_rt)

            now_point = x_rt[torch.arange(self.batch_size),idx,:]#(batch,embedding_size)
            graph_info = torch.cat((now_point,ave_x_rt),dim=1)#(batch,2*embedding_size)
            q = self.dc_wq(graph_info)
            k = self.dc_wk(x_rt)
            v = self.dc_wv(x_rt)
            q = q.reshape(self.batch_size,1,self.attention_head,self.dk)
            k = k.reshape(self.batch_size,self.rt_n, self.attention_head, self.dk)
            v = v.reshape(self.batch_size,self.rt_n, self.attention_head, self.dk)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 3, 1)
            v = v.permute(0, 2, 1, 3)
            qk = torch.matmul(q, k) / (self.dk ** 0.5)
            #mask
            qk.masked_fill_(mask_,-float('inf'))
            qk = F.softmax(qk, dim=-1) #(batch, attention_head, 1, n_rt)
            z = torch.matmul(qk, v) #(batch, attention_head, 1, dk)
            z = z.permute(0, 2, 1, 3) #(batch, 1, attention_head, dk)
            z = z.reshape(self.batch_size, 1, self.embedding_size)
            z = self.dc_w(z) #(batch, 1, embedding_size)

            #输出概率
            q_out = self.out_wq(z) #(batch, 1, embedding_size)
            k_out = self.out_wk(x_rt)
            k_out = k_out.permute(0, 2, 1)#(batch, embedding_size, n_rt)
            qk_out = torch.matmul(q_out, k_out) / (self.embedding_size ** 0.5)
            qk_out = qk_out.sum(dim=1) #(batch, n_rt)
            qk_out = torch.tanh(qk_out)*self.C
            qk_out.masked_fill_(mask,-float('inf'))
            p = F.softmax(qk_out, dim=-1)

            # 检查 p 张量中的异常值
            if torch.any(torch.isnan(p)) or torch.any(torch.isinf(p)) or torch.any(p < 0):
                print("qk_out contains invalid values:")
                # print(x_rt_debug)
                print(x_rt_debug.shape)
                if torch.any(torch.isnan(x_rt_debug)):
                    print("nan")
                    k = torch.isnan(x_rt_debug)
                    print(k)
                # print(k)
                raise ValueError("p tensor contains either `inf`, `nan` or element < 0")

            if is_train:
                idx = torch.multinomial(p,1).squeeze()
            else:
                idx = torch.argmax(p,dim=1)

            #计算距离
            pro[:,i] = p[torch.arange(self.batch_size),idx]
            seq[:,i] = idx
            distance = distance + costmap[torch.arange(self.batch_size),idx_last,idx] #索引为列表时，返回的是列表对应位置的元素组成的列表
            idx_last = idx

        if is_train==False:
            seq = seq.detach()
            pro = pro.detach()
            distance = distance.detach()
        
        return seq, pro, distance

        

    def config(self,cfg:dict):
        self.robot_n = int(cfg['n_robot'])
        self.task_n = int(cfg['n_task'])
        self.ob_n = int(cfg['n_obstacle'])
        self.rt_n = self.robot_n + self.task_n
        self.global_points_n = self.rt_n + self.ob_n
        self.ob_points = int(cfg['ob_points'])
        self.batch_size = int(cfg['batch_size'])

        
        

        