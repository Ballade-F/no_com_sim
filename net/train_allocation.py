import torch
import os
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from allocation import AllocationNet
from dataset_allocation import AllocationDataset


if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print('Using GPU')
else:
    DEVICE = torch.device('cpu')
    print('Using CPU')

def train_allocation_net():
    # configuration
    embedding_size = 128
    attention_head = 8
    num_epochs = 10
    learning_rate = 0.0002
    save_dir = '/home/zj/Desktop/wzr/no_com_sim/net_model/allocation/'
    dataset_dir = "/home/zj/Desktop/wzr/no_com_sim/allocation_data/"
    n_batch = 10
    C=10
    is_train = True

    bl_alpha = 0.05  # 做t-检验更新baseline时所设置的阈值
    test2save_times = 20  # 训练过程中每次保存模型所需的测试batch数
    min = 1000  # 当前已保存的所有模型中测试路径长度的最小值

    # Load dataset 
    dataset = AllocationDataset(dataset_dir, n_batch)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    n_ob_points = dataset.ob_points
    batch_size = dataset.batch_size
    

    # Initialize model, loss function, and optimizer
    model_train = AllocationNet(n_ob_points, embedding_size, batch_size, attention_head, C=C,device=DEVICE).to(DEVICE)
    # model_target = AllocationNet(n_ob_points, embedding_size, batch_size, attention_head, C=C,device=DEVICE).to(DEVICE)


    optimizer = optim.Adam(model_train.parameters(), lr=learning_rate)


    # Training loop
    for epoch in range(num_epochs):
        model_train.train()
        running_loss = 0.0
        for i, (feature_robot, feature_task, feature_obstacle, costmats, cfg) in enumerate(dataloader):
            feature_robot, feature_task, feature_obstacle, costmats = feature_robot.to(DEVICE), feature_task.to(DEVICE), feature_obstacle.to(DEVICE), costmats.to(DEVICE)
            feature_robot = feature_robot.squeeze(0)
            feature_task = feature_task.squeeze(0)
            feature_obstacle = feature_obstacle.squeeze(0)
            costmats = costmats.squeeze(0)

            #debug
            print(i)
            print(feature_robot.shape)
            print(feature_task.shape)
            print(feature_obstacle.shape)
            print(costmats.shape)
            

            # Forward pass
            model_train.config(cfg)
            seq, pro, distance = model_train(feature_robot, feature_task, feature_obstacle, costmats, is_train=True)

            pro_log = torch.log(pro)#(batch, n-1)
            pro_sum = torch.sum(pro_log, dim=1) #(batch)
            dis = distance.detach()
            loss = dis * pro_sum
            loss = torch.sum(loss) / batch_size

            #debug
            print(loss)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model_train.parameters(), 1)
            optimizer.step()

        # Print statistics
        print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}")
        running_loss = 0.0

    # Save model
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model_train.state_dict(), os.path.join(save_dir, "time_{}_dis_.pt".format
                                                      (time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))))
    
    print('Finished Training')

if __name__ == '__main__':
    train_allocation_net()
