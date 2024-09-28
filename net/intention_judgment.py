import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


# DEVICE = torch.device('cpu')

class IntentionNet(nn.Module):
    def __init__(self, embedding_size:int, robot_n:int,task_n:int, batch_size, attention_head:int):
        super(IntentionNet, self).__init__()
        self.embedding_size = embedding_size
        self.robot_n = robot_n
        self.task_n = task_n + 1 # 最后一个task是虚拟task，用于结束任务的robot
        self.batch_size = batch_size
        self.attention_head = attention_head
        self.dk = int(embedding_size / attention_head)
        # x,y in 5 time steps
        self.feature_robot = 10
        # x,y,finish_flag
        self.feature_task = 3

        # 嵌入
        self.embedding_robot = nn.Linear(self.feature_robot, embedding_size)
        self.embedding_task = nn.Linear(self.feature_task, embedding_size)

        # encoder
        # TODO:R和T的各个W矩阵有必要分开吗？有没有可能不收敛
        # TODO:正则化
        self.wq_r1 = nn.Linear(embedding_size, embedding_size)
        self.wk_r1 = nn.Linear(embedding_size, embedding_size)
        self.wv_r1 = nn.Linear(embedding_size, embedding_size)
        self.w_r1 = nn.Linear(embedding_size, embedding_size)
        self.wq_t1 = nn.Linear(embedding_size, embedding_size)
        self.wk_t1 = nn.Linear(embedding_size, embedding_size)
        self.wv_t1 = nn.Linear(embedding_size, embedding_size)
        self.w_t1 = nn.Linear(embedding_size, embedding_size)
        self.ffc11 = nn.Linear(embedding_size, embedding_size)
        self.ffc12 = nn.Linear(embedding_size, embedding_size)
        self.bn11 = nn.BatchNorm1d(embedding_size)
        self.bn12 = nn.BatchNorm1d(embedding_size)

        self.wq_r2 = nn.Linear(embedding_size, embedding_size)
        self.wk_r2 = nn.Linear(embedding_size, embedding_size)
        self.wv_r2 = nn.Linear(embedding_size, embedding_size)
        self.w_r2 = nn.Linear(embedding_size, embedding_size)
        self.wq_t2 = nn.Linear(embedding_size, embedding_size)
        self.wk_t2 = nn.Linear(embedding_size, embedding_size)
        self.wv_t2 = nn.Linear(embedding_size, embedding_size)
        self.w_t2 = nn.Linear(embedding_size, embedding_size)
        self.ffc21 = nn.Linear(embedding_size, embedding_size)
        self.ffc22 = nn.Linear(embedding_size, embedding_size)
        self.bn21 = nn.BatchNorm1d(embedding_size)
        self.bn22 = nn.BatchNorm1d(embedding_size)

        self.wq_r3 = nn.Linear(embedding_size, embedding_size)
        self.wk_r3 = nn.Linear(embedding_size, embedding_size)
        self.wv_r3 = nn.Linear(embedding_size, embedding_size)
        self.w_r3 = nn.Linear(embedding_size, embedding_size)
        self.wq_t3 = nn.Linear(embedding_size, embedding_size)
        self.wk_t3 = nn.Linear(embedding_size, embedding_size)
        self.wv_t3 = nn.Linear(embedding_size, embedding_size)
        self.w_t3 = nn.Linear(embedding_size, embedding_size)
        self.ffc31 = nn.Linear(embedding_size, embedding_size)
        self.ffc32 = nn.Linear(embedding_size, embedding_size)
        self.bn31 = nn.BatchNorm1d(embedding_size)
        self.bn32 = nn.BatchNorm1d(embedding_size)

        #decoder
        # TODO: 试试用mask盖住已完成的
        self.wq_rd = nn.Linear(embedding_size, embedding_size)
        self.wk_td = nn.Linear(embedding_size, embedding_size)


    
    def attention(self, x_r,x_t, wq_r, wk_r, wv_r, w_r, wq_t, wk_t, wv_t, w_t, add_residual):
        '''
        x_r: (batch, robot_n, embedding_size)   
        x_t: (batch, self.task_n, embedding_size)        
        '''
        q_r = wq_r(x_r)
        k_r = wk_r(x_r)
        v_r = wv_r(x_r)
        q_t = wq_t(x_t)
        k_t = wk_t(x_t)
        v_t = wv_t(x_t)
        q_r = q_r.contiguous().view(self.batch_size, self.robot_n, self.attention_head, self.dk)
        k_r = k_r.contiguous().view(self.batch_size, self.robot_n, self.attention_head, self.dk)
        v_r = v_r.contiguous().view(self.batch_size, self.robot_n, self.attention_head, self.dk)
        q_r = q_r.permute(0, 2, 1, 3) #(batch, attention_head, robot_n, dk)
        k_r = k_r.permute(0, 2, 3, 1) #(batch, attention_head, dk, robot_n)
        v_r = v_r.permute(0, 2, 1, 3) #(batch, attention_head, robot_n, dk)
        q_t = q_t.contiguous().view(self.batch_size, self.task_n, self.attention_head, self.dk)
        k_t = k_t.contiguous().view(self.batch_size, self.task_n, self.attention_head, self.dk)
        v_t = v_t.contiguous().view(self.batch_size, self.task_n, self.attention_head, self.dk)
        q_t = q_t.permute(0, 2, 1, 3) #(batch, attention_head, task_n, dk)
        k_t = k_t.permute(0, 2, 3, 1) #(batch, attention_head, dk, task_n)
        v_t = v_t.permute(0, 2, 1, 3) #(batch, attention_head, task_n, dk)

        k = torch.cat((k_r, k_t), dim=3) #(batch, attention_head, dk, robot_n+task_n)
        v = torch.cat((v_r, v_t), dim=2) #(batch, attention_head, robot_n+task_n, dk)
        qk_r = torch.matmul(q_r, k) / (self.dk ** 0.5) #(batch, attention_head, robot_n, robot_n+task_n)
        qk_t = torch.matmul(q_t, k_t) / (self.dk ** 0.5) #(batch, attention_head, task_n, task_n)
        qk_r = F.softmax(qk_r, dim=-1)
        qk_t = F.softmax(qk_t, dim=-1)

        z_r = torch.matmul(qk_r, v) #(batch, attention_head, robot_n, dk)
        z_r = z_r.permute(0, 2, 1, 3) #(batch, robot_n, attention_head, dk)
        z_r = z_r.contiguous().view(self.batch_size, self.robot_n, self.embedding_size)
        z_r = w_r(z_r)
        z_t = torch.matmul(qk_t, v_t) #(batch, attention_head, task_n, dk)
        z_t = z_t.permute(0, 2, 1, 3) #(batch, task_n, attention_head, dk)
        z_t = z_t.contiguous().view(self.batch_size, self.task_n, self.embedding_size)
        z_t = w_t(z_t)

        if add_residual:
            z_r = z_r + x_r
            z_t = z_t + x_t
        return z_r, z_t


    def forward(self,x_r_,x_t_,is_train):
         # 嵌入层
        x_r = self.embedding_robot(x_r_)
        x_t = self.embedding_task(x_t_)
        # encoder
        # 第一层
        x_r,x_t = self.attention(x_r,x_t,self.wq_r1,self.wk_r1,self.wv_r1,self.w_r1,self.wq_t1,self.wk_t1,self.wv_t1,self.w_t1, add_residual=True)
        x = torch.cat((x_r, x_t), dim=1)#(batch, robot_n+task_n, embedding_size)
        x = self.bn11(x.permute(0, 2, 1)).permute(0, 2, 1)#BatchNorm1d对二维中的最后一维，或三维中的中间一维进行归一化
        x1 = self.ffc11(x)
        x1 = F.relu(x1)
        x1 = self.ffc12(x1)
        x = x1 + x
        x = self.bn12(x.permute(0, 2, 1)).permute(0, 2, 1)
        x_r = x[:, :self.robot_n, :]
        x_t = x[:, self.robot_n:, :]
        # 第二层
        x_r,x_t = self.attention(x_r,x_t,self.wq_r2,self.wk_r2,self.wv_r2,self.w_r2,self.wq_t2,self.wk_t2,self.wv_t2,self.w_t2, add_residual=True)
        x = torch.cat((x_r, x_t), dim=1)
        x = self.bn21(x.permute(0, 2, 1)).permute(0, 2, 1)#BatchNorm1d对二维中的最后一维，或三维中的中间一维进行归一化
        x1 = self.ffc21(x)
        x1 = F.relu(x1)
        x1 = self.ffc22(x1)
        x = x1 + x
        x = self.bn22(x.permute(0, 2, 1)).permute(0, 2, 1)
        x_r = x[:, :self.robot_n, :]
        x_t = x[:, self.robot_n:, :]
        # 第三层
        x_r,x_t = self.attention(x_r,x_t,self.wq_r3,self.wk_r3,self.wv_r3,self.w_r3,self.wq_t3,self.wk_t3,self.wv_t3,self.w_t3, add_residual=True)
        x = torch.cat((x_r, x_t), dim=1)
        x = self.bn31(x.permute(0, 2, 1)).permute(0, 2, 1)#BatchNorm1d对二维中的最后一维，或三维中的中间一维进行归一化
        x1 = self.ffc31(x)
        x1 = F.relu(x1)
        x1 = self.ffc32(x1)
        x = x1 + x
        x = self.bn32(x.permute(0, 2, 1)).permute(0, 2, 1)
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