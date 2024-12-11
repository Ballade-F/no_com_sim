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
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train Allocation Network')
parser.add_argument('--device', type=str, default='cuda:3', help='Device to use for training (e.g., "cuda:2" or "cpu")')
parser.add_argument('--lr', type=float, default=0.00012, help='Learning rate for training')
parser.add_argument('--lr_step', type=int, default=50, help='Number of epochs before reducing learning rate')
parser.add_argument('--lr_gamma', type=float, default=0.95, help='Multiplicative factor for reducing learning rate')
parser.add_argument('--embedding_size', type=int, default=128, help='Size of the embedding vector')
parser.add_argument('--attention_head', type=int, default=8, help='Number of attention heads')
parser.add_argument('--n_batch', type=int, default=1024, help='Number of batches in the training dataset')
parser.add_argument('--test_batch', type=int, default=20, help='Number of batches in the test dataset')
parser.add_argument('--n_epoch', type=int, default=500, help='Number of epochs to train')
parser.add_argument('--save_dir', type=str, default="allocation", help='Directory to save the model')
parser.add_argument('--log_dir', type=str, default="allocation", help='Directory to save the log')
parser.add_argument('--data_dir', type=str, default="allocation", help='Directory of the training dataset')
parser.add_argument('--test_dir', type=str, default="allocation_test", help='Directory of the test dataset')

args = parser.parse_args()

# Set device
if args.device.startswith('cuda') and torch.cuda.is_available():
    DEVICE = torch.device(args.device)
    print('Using GPU')
else:
    DEVICE = torch.device('cpu')
    print('Using CPU')

# if torch.cuda.is_available():
#     DEVICE = torch.device('cuda:2')
#     print('Using GPU')
# else:
#     DEVICE = torch.device('cpu')
#     print('Using CPU')


def train_allocation_net():
    # configuration
    embedding_size = args.embedding_size
    attention_head = args.attention_head
    num_epochs = args.n_epoch
    learning_rate = args.lr
    lr_step = args.lr_step
    lr_gamma = args.lr_gamma
    save_dir_root = '/home/data/wzr/no_com_1/model/allocation'
    dataset_dir_root = "/home/data/wzr/no_com_1/data"
    dataset_test_dir_root = "/home/data/wzr/no_com_1/data"
    save_dir = os.path.join(save_dir_root, args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    dataset_dir = os.path.join(dataset_dir_root, args.data_dir)
    test_dir = os.path.join(dataset_test_dir_root, args.test_dir)
    
    n_batch = args.n_batch
    test_batch = args.test_batch
    C=10
    is_train = True

    bl_alpha = 0.05  # 做t-检验更新baseline时所设置的阈值
    min = 36.5  # 当前已保存的所有模型中测试路径长度的最小值

    # Load dataset 
    dataset = AllocationDataset(dataset_dir, n_batch)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    dataset_test = AllocationDataset(test_dir, test_batch)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=True)

    n_ob_points = dataset.ob_points
    batch_size = dataset.batch_size
    # batch_size = 1
    

    # Initialize model, loss function, and optimizer
    model_train = AllocationNet(n_ob_points, embedding_size, batch_size, attention_head, C=C,device=DEVICE).to(DEVICE)
    model_target = AllocationNet(n_ob_points, embedding_size, batch_size, attention_head, C=C,device=DEVICE).to(DEVICE)

    optimizer = optim.Adam(model_train.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    # Training loop
    for epoch in range(num_epochs):
        model_train.train()
        model_train.is_train = True
        model_target.is_train = False
        running_loss = 0.0
        for i, (feature_robot, feature_task, feature_obstacle, costmats, cfg) in enumerate(dataloader):
            feature_robot, feature_task, feature_obstacle, costmats = feature_robot.to(DEVICE), feature_task.to(DEVICE), feature_obstacle.to(DEVICE), costmats.to(DEVICE)
            feature_robot = feature_robot.squeeze(0)
            feature_task = feature_task.squeeze(0)
            feature_obstacle = feature_obstacle.squeeze(0)
            costmats = costmats.squeeze(0)
            # Forward pass
            model_train.config(cfg)
            model_target.config(cfg)
            # time_start = time.time()
            seq, pro, distance = model_train(feature_robot, feature_task, feature_obstacle, costmats)
            # time_end = time.time()
            # print('time:', time_end-time_start)
            seq_target, pro_target, distance_target = model_target(feature_robot, feature_task, feature_obstacle, costmats)#baseline

            pro_log = torch.log(pro)#(batch, n-1)
            loss = torch.sum(pro_log, dim=1) #(batch)
            score = distance-distance_target
            score = score.detach()
            loss = score * loss
            loss = torch.sum(loss) / batch_size
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model_train.parameters(), 1)
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], batch [{i + 1}/{len(dataloader)}], Loss: {loss.detach().item():.4f}, Distance: {distance.mean().item():.4f}')
                logging.info(f'Epoch [{epoch + 1}/{num_epochs}], batch [{i + 1}/{len(dataloader)}], Loss: {loss.detach().item():.4f}, Distance: {distance.mean().item():.4f}')
                    
            # OneSidedPairedTTest(做t-检验看当前Sampling的解效果是否显著好于greedy的解效果,如果是则更新使用greedy策略作为baseline的net2参数)
            if (distance.mean() - distance_target.mean()) < 0:
                tt, pp = ttest_rel(distance.cpu().numpy(), distance_target.cpu().numpy())
                p_val = pp / 2
                assert tt < 0, "T-statistic should be negative"
                if p_val < bl_alpha:
                    print('Update baseline')
                    logging.info('Update baseline')
                    model_target.load_state_dict(model_train.state_dict())

            # 每隔xxx步做测试判断结果有没有改进，如果改进了则把当前模型保存下来
            if (i+1) % 100 == 0:
                model_train.is_train = False
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
                        seq, pro, distance = model_train(feature_robot, feature_task, feature_obstacle, costmats)
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
                model_train.is_train = True
                model_train.train()

        scheduler.step()  
        print(f'Epoch [{epoch + 1}/{num_epochs}], Learning rate: {scheduler.get_last_lr()[0]}')  
        logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Learning rate: {scheduler.get_last_lr()[0]}')  
        
    # # Save model
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # torch.save(model_train.state_dict(), os.path.join(save_dir, "time_{}_dis_.pt".format
    #                                                   (time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))))
    


if __name__ == '__main__':
    log_dir_root = '/home/data/wzr/no_com_1/log'
    log_dir = args.log_dir
    log_dir = os.path.join(log_dir_root, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"train_allocation_{TM.strftime('%Y-%m-%d-%H-%M', TM.localtime())}.log")
    logging.basicConfig(filename=log_path,
                         level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logging.info('BEGIN')
    logging.info('Device: {}'.format(DEVICE))
    logging.info('Learning rate: {}'.format(args.lr))
    logging.info('Learning rate step: {}'.format(args.lr_step))
    logging.info('Learning rate gamma: {}'.format(args.lr_gamma))
    logging.info('Embedding size: {}'.format(args.embedding_size))
    logging.info('Attention head: {}'.format(args.attention_head))
    logging.info('Number of batches: {}'.format(args.n_batch))
    logging.info('Number of test batches: {}'.format(args.test_batch))
    logging.info('Number of epochs: {}'.format(args.n_epoch))
    logging.info('Save directory: {}'.format(args.save_dir))
    logging.info('Log directory: {}'.format(args.log_dir))
    logging.info('Data directory: {}'.format(args.data_dir))
    logging.info('Test directory: {}'.format(args.test_dir))
    
    train_allocation_net()
    print('Finished Training')
    logging.info('Finished Training')
