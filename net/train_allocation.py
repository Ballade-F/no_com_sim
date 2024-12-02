import torch
import os
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from allocation import AllocationNet
from dataset_allocation import AllocationDataset
from scipy.stats import ttest_rel
import time as TM
import logging

# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu' )


if torch.cuda.is_available():
    DEVICE = torch.device('cuda:1')
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
    learning_rate = 0.0005
    save_dir = '/home/data/wzr/no_com_1/model/allocation'
    dataset_dir = "/home/data/wzr/no_com_1/data/allocation_2024"
    test_dir = "/home/data/wzr/no_com_1/data/allocation_2024_test"
    n_batch = 10
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
            # time_start = time.time()
            seq, pro, distance = model_train(feature_robot, feature_task, feature_obstacle, costmats, is_train=True)
            # time_end = time.time()
            # print('time:', time_end-time_start)
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
            
            print(f'Epoch [{epoch + 1}/{num_epochs}], batch [{i + 1}/{len(dataloader)}], Loss: {loss.detach().item():.4f}, Distance: {distance.mean().item():.4f}')
            logging.info(f'Epoch [{epoch + 1}/{num_epochs}], batch [{i + 1}/{len(dataloader)}], Loss: {loss.detach().item():.4f}')
                    

            # OneSidedPairedTTest(做t-检验看当前Sampling的解效果是否显著好于greedy的解效果,如果是则更新使用greedy策略作为baseline的net2参数)
            if (distance.mean() - distance_target.mean()) < 0:
                tt, pp = ttest_rel(distance.cpu().numpy(), distance_target.cpu().numpy())
                p_val = pp / 2
                assert tt < 0, "T-statistic should be negative"
                if p_val < bl_alpha:
                    print('Update baseline')
                    model_target.load_state_dict(model_train.state_dict())

            # 每隔xxx步做测试判断结果有没有改进，如果改进了则把当前模型保存下来
            #测试集
            if (i+1) % 50 == 0:
                model_train.eval()
                with torch.no_grad():
                    time_ave = 0.0
                    length = 0.0
                    for _, (feature_robot, feature_task, feature_obstacle, costmats, cfg) in enumerate(dataloader_test):
                        feature_robot, feature_task, feature_obstacle, costmats = feature_robot.to(DEVICE), feature_task.to(DEVICE), feature_obstacle.to(DEVICE), costmats.to(DEVICE)
                        feature_robot = feature_robot.squeeze(0)
                        feature_task = feature_task.squeeze(0)
                        feature_obstacle = feature_obstacle.squeeze(0)
                        costmats = costmats.squeeze(0)

                        # Forward pass
                        model_train.config(cfg)
                        #测时间
                        time_start = time.time()
                        seq, pro, distance = model_train(feature_robot, feature_task, feature_obstacle, costmats, is_train=False)
                        time_end = time.time()
                        time_ave += time_end - time_start
                        length += distance.mean().item()
                    length = length/test_batch
                    time_ave = time_ave/test_batch

                    if length < min:
                        min = length
                        torch.save(model_train.state_dict(), os.path.join(save_dir, "time_{}_dis_{:.2f}.pt".format
                                                        (time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()), min)))
                        print('Save model')
                        logging.info('Save model')

                    # Print statistics
                    print(f"TEST: Epoch {epoch}, Batch {i}, min {min}, length {length}, time {time_ave}")
                    logging.info(f"TEST: Epoch {epoch}, Batch {i}, min {min}, length {length}, time {time_ave}")
                model_train.train()


    # # Save model
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # torch.save(model_train.state_dict(), os.path.join(save_dir, "time_{}_dis_.pt".format
    #                                                   (time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))))
    


if __name__ == '__main__':
    logging.basicConfig(filename='/home/users/wzr/project/no_com_sim/log/train_allocation_{}.log'.format(TM.strftime("%Y-%m-%d-%H-%M", TM.localtime())),
                         level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logging.info('BEGIN')
    train_allocation_net()
    print('Finished Training')
    logging.info('Finished Training')
