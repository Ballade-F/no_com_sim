import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from module import Self_Attention, Self_Cross_Attention, SelfAttentionBlock



class _encoderBlock(nn.Module):
    def __init__(self, embedding_size:int, attention_head:int):
        super(_encoderBlock, self).__init__()

        self.robot_n = -1
        self.task_n = -1  # 最后一个task是虚拟task，用于结束任务的robot

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
    def __init__(self, ob_points:int = 16, r_points:int = 5,
                 embedding_size:int = 128, batch_size:int = 1, attention_head:int = 8,
                 feature_robot:int = 2, feature_task:int = 3, feature_ob:int = 2,
                 encoder_layer:int = 3, local_embed_layers:int=2, device='cpu'):
        super(IntentionNet, self).__init__()
        self.embedding_size = embedding_size
        self.robot_n = -1
        self.task_n = -1 # 最后一个task是虚拟task，用于结束任务的robot
        self.ob_n = -1

        self.batch_size = batch_size
        self.ob_points = ob_points
        self.r_points = r_points

        # 嵌入
        self.embedding_robot = nn.Linear(feature_robot, embedding_size)
        nn.init.kaiming_normal_(self.embedding_robot.weight)
        self.embedding_task = nn.Linear(feature_task, embedding_size)
        nn.init.kaiming_normal_(self.embedding_task.weight)
        self.embedding_ob = nn.Linear(feature_ob, embedding_size)
        nn.init.kaiming_normal_(self.embedding_ob.weight)

        self.robot_encoder_layers = nn.ModuleList([
            SelfAttentionBlock(embedding_size, attention_head)
            for _ in range(local_embed_layers)
        ])
        self.ob_encoder_layers = nn.ModuleList([
            SelfAttentionBlock(embedding_size, attention_head)
            for _ in range(local_embed_layers)
        ])

        # encoder
        # TODO:正则化
        self.global_encoder_layers = nn.ModuleList([
            SelfAttentionBlock(embedding_size, attention_head)
            for _ in range(encoder_layer)
        ])

        #decoder
        # TODO: 试试用mask盖住已完成的
        self.wq_rd = nn.Linear(embedding_size, embedding_size)
        nn.init.kaiming_normal_(self.wq_rd.weight)
        self.wk_td = nn.Linear(embedding_size, embedding_size)
        nn.init.kaiming_normal_(self.wk_td.weight)

#x_r: (batch, n_robot,r_points, 2), x_t: (batch, self.n_task, 3), x_ob: (batch, n_obstacle, ob_points, 2)
    def forward(self,x_r_,x_t_,x_ob_,is_train):
         # 嵌入层
        x_r = self.embedding_robot(x_r_)
        x_t = self.embedding_task(x_t_)
        x_ob = self.embedding_ob(x_ob_)

        # local embedding
        x_r = x_r.reshape(self.batch_size*self.robot_n, self.r_points, self.embedding_size)
        for encoder_layer in self.robot_encoder_layers:
            x_r = encoder_layer(x_r)
        x_r = x_r.reshape(self.batch_size, self.robot_n, self.r_points, self.embedding_size)

        if self.ob_n > 0:
            x_ob = x_ob.reshape(self.batch_size*self.ob_n, self.ob_points, self.embedding_size)
            for encoder_layer in self.ob_encoder_layers:
                x_ob = encoder_layer(x_ob)
            x_ob = x_ob.reshape(self.batch_size, self.ob_n, self.ob_points, self.embedding_size)

        x_r = torch.mean(x_r, dim=2)
        x_ob = torch.mean(x_ob, dim=2)

        # encoder
        x = torch.cat((x_r, x_t, x_ob), dim=1)
        for encoder_layer in self.global_encoder_layers:
            x = encoder_layer(x)
        x_r = x[:, :self.robot_n, :]
        x_t = x[:, self.robot_n:(self.robot_n+self.task_n), :]

        #decoder
        q_rd = self.wq_rd(x_r)
        k_td = self.wk_td(x_t)
        k_td = k_td.permute(0, 2, 1)#(batch, embedding_size, task_n)
        qk_d = torch.matmul(q_rd, k_td) / (self.embedding_size ** 0.5)#(batch, robot_n, task_n)
        # p = F.softmax(qk_d, dim=-1)

        return qk_d
    
    def config(self,cfg:dict):
        self.robot_n = int(cfg['n_robot'])
        self.task_n = int(cfg['n_task'])+1
        self.ob_n = int(cfg['n_obstacle'])
        

    

# Test function
def test_intention_judgment_model():
    batch_size = 2
    robot_n = 5
    task_n = 3
    ob_n = 2
    ob_points = 16
    r_points = 5
    embedding_size = 16
    attention_head = 4

    model = IntentionNet(ob_points, r_points, embedding_size, batch_size, attention_head)
    model.to(DEVICE)
    model.config({'n_robot':robot_n, 'n_task':task_n, 'n_obstacle':ob_n})

    x_r = torch.randn(batch_size, robot_n, r_points, 2).to(DEVICE)
    x_t = torch.randn(batch_size, task_n + 1, 3).to(DEVICE)
    x_ob = torch.randn(batch_size, ob_n, ob_points, 2).to(DEVICE)

    output = model(x_r, x_t, x_ob, is_train=True)
    print("Output Shape:", output.shape)
    assert output.shape == (batch_size, robot_n, task_n + 1)
    assert isinstance(output, torch.Tensor)

if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
    test_intention_judgment_model()