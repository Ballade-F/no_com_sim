import torch
import os
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from intention import IntentionNet
from dataset_intention import IntentionDataset
import json
import time as TM
import logging
import argparse

# if torch.cuda.is_available():
#     DEVICE = torch.device('cuda')
#     print('Using GPU')
# else:
#     DEVICE = torch.device('cpu')
#     print('Using CPU')

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train Intention Network')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training (e.g., "cuda:2" or "cpu")')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')
parser.add_argument('--lr_step', type=int, default=50, help='Number of epochs before reducing learning rate')
parser.add_argument('--lr_gamma', type=float, default=0.9, help='Multiplicative factor for reducing learning rate')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--embedding_size', type=int, default=128, help='Size of the embedding vector')
parser.add_argument('--attention_head', type=int, default=8, help='Number of attention heads')
parser.add_argument('--n_scale', type=int, default=1024, help='Number of batches in the training dataset')
parser.add_argument('--n_scale_test', type=int, default=10, help='Number of batches in the test dataset')
parser.add_argument('--n_epoch', type=int, default=20, help='Number of epochs to train')

args = parser.parse_args()

# Set device
if args.device.startswith('cuda') and torch.cuda.is_available():
    DEVICE = torch.device(args.device)
    print('Using GPU')
else:
    DEVICE = torch.device('cpu')
    print('Using CPU')


def train_intention_net():
    # Hyperparameters
    embedding_size = args.embedding_size
    batch_size = args.batch_size
    attention_head = args.attention_head
    r_points = 5
    num_epochs = args.n_epoch
    n_scale = args.n_scale
    n_scale_test = args.n_scale_test
    learning_rate = args.lr
    lr_step = args.lr_step
    lr_gamma = args.lr_gamma    

    save_dir = '/home/data/wzr/no_com_1/model/intention/2024_12_9'
    dataset_dir = "/home/data/wzr/no_com_1/data/intention_2024"
    dataset_test_dir = "/home/data/wzr/no_com_1/data/intention_2024_test"
    # pre_model = '/home/data/wzr/no_com_1/model/intention/time_2024-12-01-12-06-51_acc_93.22.pt'
    # read json file
    with open(os.path.join(dataset_dir, "dataset_info.json"), "r") as f:
        dataset_info = json.load(f)
    # n_scale = dataset_info["n_scale"]
    # n_scale = 10
    ob_points = dataset_info["ob_points"]
    
    with open(os.path.join(dataset_test_dir, "dataset_info.json"), "r") as f:
        dataset_info = json.load(f)
    # n_scale_test = dataset_info["n_scale"]
    # n_scale_test = 10
    ob_points_test = dataset_info["ob_points"]
    

    # Initialize model, loss function, and optimizer
    model = IntentionNet(ob_points, r_points, embedding_size, batch_size, attention_head, device=DEVICE).to(DEVICE)
    # model.load_state_dict(torch.load(pre_model))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    # datasets = [IntentionDataset(os.path.join(dataset_dir, f"scale_{i}"), r_points) for i in range(n_scale)]
    # dataloaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True) for dataset in datasets]
    datasets = []
    dataloaders = []
    dataset = None
    dataloader = None
    
    datasets_test = [IntentionDataset(os.path.join(dataset_test_dir, f"scale_{i}"), r_points) for i in range(n_scale_test)]
    dataloaders_test = [DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True) for dataset in datasets_test]

    min_loss = 0.1
    running_loss = 0.0
    train_count = 0
    max_acc = 96.0
    # epoch training
    for epoch in range(num_epochs):
        model.train()
        for i_scale in range(n_scale):
            if epoch == 0:
                dataset = IntentionDataset(os.path.join(dataset_dir, f"scale_{i_scale}"), r_points)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
                datasets.append(dataset)
                dataloaders.append(dataloader)
            else:
                dataset = datasets[i_scale]
                dataloader = dataloaders[i_scale]
            # dataset = datasets[i_scale]
            # dataloader = dataloaders[i_scale]    
            cfg = dataset.cfg
            model.config(cfg)
            for i, (feature_robot, label, feature_task, feature_obstacle) in enumerate(dataloader):
                feature_robot, label, feature_task, feature_obstacle = feature_robot.to(DEVICE), label.to(DEVICE), feature_task.to(DEVICE), feature_obstacle.to(DEVICE)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(feature_robot, feature_task, feature_obstacle)
                loss = criterion(outputs.reshape(-1, 1+cfg["n_task"]), label.reshape(-1).long())

                loss_detached = loss.detach().item()
                running_loss += loss_detached
                train_count += 1

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

            # Print statistics
            if (i_scale+1) % 5 == 0:
                loss_ave = running_loss / train_count
                print(f'Epoch [{epoch + 1}/{num_epochs}], Scale [{i_scale + 1}/{n_scale}], Loss: {loss_ave:.4f}')
                logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Scale [{i_scale + 1}/{n_scale}], Loss: {loss_ave:.4f}')
                running_loss = 0.0
                train_count = 0
                    
            # test
            if (i_scale+1) % 50 == 0:
                model.eval()
                with torch.no_grad():
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Learning rate: {scheduler.get_last_lr()[0]}')  
                    logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Learning rate: {scheduler.get_last_lr()[0]}')  
                    test_loss = 0.0
                    right = 0
                    error = 0
                    count = 0
                    time_ave = 0.0
                    for i_scale in range(n_scale_test):
                        dataset = datasets_test[i_scale]
                        dataloader = dataloaders_test[i_scale]
                        cfg = dataset.cfg
                        model.config(cfg)
                        for i, (feature_robot, label, feature_task, feature_obstacle) in enumerate(dataloader):
                            feature_robot, label, feature_task, feature_obstacle = feature_robot.to(DEVICE), label.to(DEVICE), feature_task.to(DEVICE), feature_obstacle.to(DEVICE)
                            #测时间
                            time_start = time.time()
                            outputs = model(feature_robot, feature_task, feature_obstacle)
                            time_end = time.time()
                            time_ave += time_end - time_start
                            model_label = torch.argmax(outputs, dim=-1)
                            #如果model_label和label相等，则right += 1,否则error += 1
                            right += torch.sum(model_label == label).item()
                            error += torch.sum(model_label != label).item()
                            loss = criterion(outputs.reshape(-1, 1+cfg["n_task"]), label.reshape(-1).long())
                            test_loss += loss.detach().item()
                            count += 1
                    
                    acc_ave = 100.0 * right / (right + error)
                    time_ave = time_ave / count
                    test_loss_ave = test_loss / count
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Test Loss: {test_loss_ave:.4f}, Test Accuracy: {acc_ave:.2f}%, Time: {time_ave:.4f}')
                    logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Test Loss: {test_loss_ave:.4f}, Test Accuracy: {acc_ave:.2f}%, Time: {time_ave:.4f}')
                            
                    if acc_ave > max_acc:
                        max_acc = acc_ave
                        torch.save(model.state_dict(), os.path.join(save_dir, "time_{}_acc_{:.2f}.pt".format
                                                        (time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()), acc_ave)))
                        print('Model saved')
                        logging.info('Model saved')
                model.train()
    
            scheduler.step()   
                 

    print('Finished Training')
    logging.info('Finished Training')

if __name__ == "__main__":
    logging.basicConfig(filename='/home/data/wzr/no_com_1/log/2024_12_8/train_intention_{}.log'.format(TM.strftime("%Y-%m-%d-%H-%M", TM.localtime())),
                         level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logging.info('BEGIN')
    logging.info('Device: {}'.format(DEVICE))
    logging.info('Learning rate: {}'.format(args.lr))
    logging.info('Learning rate step: {}'.format(args.lr_step))
    logging.info('Learning rate gamma: {}'.format(args.lr_gamma))
    logging.info('Batch size: {}'.format(args.batch_size))
    logging.info('Embedding size: {}'.format(args.embedding_size))
    logging.info('Attention head: {}'.format(args.attention_head))
    logging.info('Number of batches: {}'.format(args.n_scale))
    logging.info('Number of test batches: {}'.format(args.n_scale_test))
    logging.info('Number of epochs: {}'.format(args.n_epoch))
    train_intention_net()
    print('Finished Training')
    logging.info('Finished Training')