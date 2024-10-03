import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


# DEVICE = torch.device('cpu')


#
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
        nn.init.kaiming_normal_(self.wk.weight)
        self.wk_b = nn.Linear(embedding_size, embedding_size)
        nn.init.kaiming_normal_(self.wk.weight)
        self.wv_a = nn.Linear(embedding_size, embedding_size)
        nn.init.kaiming_normal_(self.wv.weight)
        self.wv_b = nn.Linear(embedding_size, embedding_size)
        nn.init.kaiming_normal_(self.wv.weight)
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

class EncoderBlock(nn.Module):
    def __init__(self, embedding_size:int, robot_n:int,task_n:int, batch_size:int, attention_head:int):
        super(EncoderBlock, self).__init__()

        self.robot_n = robot_n
        self.task_n = task_n + 1 # 最后一个task是虚拟task，用于结束任务的robot

        self.robot_attention = Self_Attention(embedding_size, attention_head)
        self.task_attention = Self_Attention(embedding_size, attention_head)
        self.bn1 = nn.BatchNorm1d(embedding_size)
        self.bn2 = nn.BatchNorm1d(embedding_size)
        self.ffc1 = nn.Linear(embedding_size, embedding_size)
        self.ffc2 = nn.Linear(embedding_size, embedding_size)
    
    # robot 在前，task 在后
    def forward(self, x):
        x_r = x[:, :self.robot_n, :]
        x_t = x[:, self.robot_n:, :]
        x_r = self.robot_attention(x_r)
        x_t = self.task_attention(x_t)
        x = torch.cat((x_r, x_t), dim=1) #(batch, robot_n+task_n, embedding_size)
        x = self.bn1(x.permute(0, 2, 1)).permute(0, 2, 1) #BatchNorm1d对二维中的最后一维，或三维中的中间一维进行归一化
        x1 = self.ffc1(x)
        x1 = F.relu(x1)
        x1 = self.ffc2(x1)
        x = x1 + x
        x = self.bn2(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x
        

class IntentionNet(nn.Module):
    def __init__(self, embedding_size:int, robot_n:int,task_n:int, batch_size, attention_head:int,
                 feature_robot:int = 10, feature_task:int = 3, encoder_layer:int = 3):
        super(IntentionNet, self).__init__()
        self.embedding_size = embedding_size
        self.robot_n = robot_n
        self.task_n = task_n + 1 # 最后一个task是虚拟task，用于结束任务的robot
        self.batch_size = batch_size
        

        # 嵌入
        self.embedding_robot = nn.Linear(feature_robot, embedding_size)
        self.embedding_task = nn.Linear(feature_task, embedding_size)

        # encoder
        # TODO:正则化
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(embedding_size, 
                         robot_n, 
                         task_n, 
                         batch_size, 
                         attention_head)
                         for _ in range(encoder_layer)])

        #decoder
        # TODO: 试试用mask盖住已完成的
        self.wq_rd = nn.Linear(embedding_size, embedding_size)
        self.wk_td = nn.Linear(embedding_size, embedding_size)


    def forward(self,x_r_,x_t_,is_train):
         # 嵌入层
        x_r = self.embedding_robot(x_r_)
        x_t = self.embedding_task(x_t_)
        # encoder
        # 第一层
        x = torch.cat((x_r, x_t), dim=1)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        x_r = x[:, :self.robot_n, :]
        x_t = x[:, self.robot_n:, :]

        #decoder
        q_rd = self.wq_rd(x_r)
        k_td = self.wk_td(x_t)
        k_td = k_td.permute(0, 2, 1)#(batch, embedding_size, task_n)
        qk_d = torch.matmul(q_rd, k_td) / (self.embedding_size ** 0.5)#(batch, robot_n, task_n)
        # p = F.softmax(qk_d, dim=-1)

        return qk_d
    

# Test function
def test_intention_judgment_model():
    batch_size = 2
    robot_n = 5
    task_n = 3
    embedding_size = 16
    attention_head = 4

    model = IntentionNet(embedding_size, robot_n, task_n, batch_size, attention_head)
    model.to(DEVICE)

    x_r = torch.randn(batch_size, robot_n, 10).to(DEVICE)
    x_t = torch.randn(batch_size, task_n + 1, 3).to(DEVICE)

    output = model(x_r, x_t, is_train=True)
    print("Output Shape:", output.shape)
    assert output.shape == (batch_size, robot_n, task_n + 1)
    assert isinstance(output, torch.Tensor)

if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
    test_intention_judgment_model()