import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from local_embed import Self_Attention, Self_Cross_Attention



class EncoderBlock(nn.Module):
    def __init__(self, embedding_size:int, robot_n:int,task_n:int, batch_size:int, attention_head:int):
        super(EncoderBlock, self).__init__()

        self.robot_n = robot_n
        self.task_n = task_n + 1 # 最后一个task是虚拟task，用于结束任务的robot

        self.robot_attention = Self_Cross_Attention(embedding_size, attention_head)
        self.task_attention = Self_Attention(embedding_size, attention_head)
        self.bn1 = nn.BatchNorm1d(embedding_size)
        self.bn2 = nn.BatchNorm1d(embedding_size)
        self.ffc1 = nn.Linear(embedding_size, embedding_size)
        self.ffc2 = nn.Linear(embedding_size, embedding_size)
    
    # robot 在前，task 在后
    def forward(self, x):
        x_r = x[:, :self.robot_n, :]
        x_t = x[:, self.robot_n:, :]
        x_r = self.robot_attention(x_r,x_t)
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