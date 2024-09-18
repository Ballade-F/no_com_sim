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

    
    def attention(self, x_r,x_t, wq_r, wk_r, wv_r, w_r, wq_t, wk_t, wv_t, w_t, add_residual):
        '''
        x_r: (batch, robot_n, embedding_size)   
        x_t: (batch, task_n, embedding_size)        
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
        z_t = torch.matmul(qk_t, v_t) #(batch, attention_head, task_n, dk)
        z_t = z_t.permute(0, 2, 1, 3) #(batch, task_n, attention_head, dk)
        z_t = z_t.contiguous().view(self.batch_size, self.task_n, self.embedding_size)z_r)
        z_t = w_t(z_t)

        if add_residual:
            z_r = z_r + x_r
            z_t = z_t + x_t
        return z_r, z_t


