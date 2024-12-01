import torch
import os
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from allocation import AllocationNet
from dataset_allocation import AllocationDataset
from scipy.stats import ttest_rel
import logging

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print('Using GPU')
else:
    DEVICE = torch.device('cpu')
    print('Using CPU')

# DEVICE = torch.device('cpu')

def train_allocation_net():
    # configuration
    embedding_size = 128
    attention_head = 8
    num_epochs = 100
    learning_rate = 0.0002
    save_dir = '/home/data/wzr/no_com_1/model/allocation'
    dataset_dir = "/home/data/wzr/no_com_1/data/allocation"
    test_dir = "/home/data/wzr/no_com_1/data/allocation_test"
    n_batch = 128
    test_batch = 1
    C=10
    is_train = True

    bl_alpha = 0.05  # 做t-检验更新baseline时所设置的阈值
    min = 100  # 当前已保存的所有模型中测试路径长度的最小值

    # Load dataset 
    dataset = AllocationDataset(dataset_dir, n_batch)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    dataset_test = AllocationDataset(test_dir, test_batch)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    n_ob_points = dataset.ob_points
    batch_size = dataset.batch_size
    # batch_size = 1
    

    # Initialize model, loss function, and optimizer
    model_train = AllocationNet(n_ob_points, embedding_size, batch_size, attention_head, C=C,device=DEVICE).to(DEVICE)
    model_target = AllocationNet(n_ob_points, embedding_size, batch_size, attention_head, C=C,device=DEVICE).to(DEVICE)


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

            # #debug
            # print("epoch", epoch, "batch", i)
            # print(feature_robot.shape)
            # print(feature_task.shape)
            # print(feature_obstacle.shape)
            # print(costmats.shape)
            

            # Forward pass
            model_train.config(cfg)
            model_target.config(cfg)
            seq, pro, distance = model_train(feature_robot, feature_task, feature_obstacle, costmats, is_train=True)
            seq_target, pro_target, distance_target = model_target(feature_robot, feature_task, feature_obstacle, costmats, is_train=False)#baseline

            pro_log = torch.log(pro)#(batch, n-1)
            loss = torch.sum(pro_log, dim=1) #(batch)
            score = distance-distance_target
            score = score.detach()
            loss = score * loss
            loss = torch.sum(loss) / batch_size

            # #debug
            # print(score)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model_train.parameters(), 1)
            optimizer.step()

            # OneSidedPairedTTest(做t-检验看当前Sampling的解效果是否显著好于greedy的解效果,如果是则更新使用greedy策略作为baseline的net2参数)
            if (distance.mean() - distance_target.mean()) < 0:
                tt, pp = ttest_rel(distance.cpu().numpy(), distance_target.cpu().numpy())
                p_val = pp / 2
                assert tt < 0, "T-statistic should be negative"
                if p_val < bl_alpha:
                    print('Update baseline')
                    model_target.load_state_dict(model_train.state_dict())

            # 每隔xxx步做测试判断结果有没有改进，如果改进了则把当前模型保存下来
            #TODO：用测试集
            if (i+1) % 30 == 0:
                model_train.eval()
                length = 0.0
                for _, (feature_robot, feature_task, feature_obstacle, costmats, cfg) in enumerate(dataloader_test):
                    feature_robot, feature_task, feature_obstacle, costmats = feature_robot.to(DEVICE), feature_task.to(DEVICE), feature_obstacle.to(DEVICE), costmats.to(DEVICE)
                    feature_robot = feature_robot.squeeze(0)
                    feature_task = feature_task.squeeze(0)
                    feature_obstacle = feature_obstacle.squeeze(0)
                    costmats = costmats.squeeze(0)

                    # Forward pass
                    model_train.config(cfg)
                    seq, pro, distance = model_train(feature_robot, feature_task, feature_obstacle, costmats, is_train=False)
                    length += distance.mean().item()
                length = length/test_batch

                if length < min:
                    min = length
                    torch.save(model_train.state_dict(), os.path.join(save_dir, "time_{}_dis_{:.2f}.pt".format
                                                      (time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()), min)))
                    print('Save model')

                # Print statistics
                print(f"Epoch {epoch}, Batch {i}, min {min}, length {length}")

                model_train.train()


    # # Save model
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # torch.save(model_train.state_dict(), os.path.join(save_dir, "time_{}_dis_.pt".format
    #                                                   (time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))))
    


if __name__ == '__main__':
    train_allocation_net()
