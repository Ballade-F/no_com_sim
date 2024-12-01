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

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print('Using GPU')
else:
    DEVICE = torch.device('cpu')
    print('Using CPU')
# DEVICE = torch.device('cpu')


def train_intention_net():
    # Hyperparameters
    embedding_size = 128
    batch_size = 128
    attention_head = 8
    r_points = 5
    num_epochs = 100
    learning_rate = 0.001

    save_dir = '/home/data/wzr/no_com_1/model/intention'
    dataset_dir = "/home/data/wzr/no_com_1/data/intention_2024"
    dataset_test_dir = "/home/data/wzr/no_com_1/data/intention_2024_test"
    # read json file
    with open(os.path.join(dataset_dir, "dataset_info.json"), "r") as f:
        dataset_info = json.load(f)
    n_scale = dataset_info["n_scale"]
    ob_points = dataset_info["ob_points"]
    
    with open(os.path.join(dataset_test_dir, "dataset_info.json"), "r") as f:
        dataset_info = json.load(f)
    # n_scale_test = dataset_info["n_scale"]
    n_scale_test = 10
    ob_points_test = dataset_info["ob_points"]
    

    # Initialize model, loss function, and optimizer
    model = IntentionNet(ob_points, r_points, embedding_size, batch_size, attention_head, device=DEVICE).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    datasets = [IntentionDataset(os.path.join(dataset_dir, f"scale_{i}"), r_points) for i in range(n_scale)]
    dataloaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True) for dataset in datasets]
    
    datasets_test = [IntentionDataset(os.path.join(dataset_test_dir, f"scale_{i}"), r_points) for i in range(n_scale_test)]
    dataloaders_test = [DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True) for dataset in datasets_test]

    min_loss = 0.1
    running_loss = 0.0
    max_acc = 0.0
    # epoch training
    for epoch in range(num_epochs):
        model.train()
        for i_scale in range(n_scale):
            dataset = datasets[i_scale]
            dataloader = dataloaders[i_scale]    
            cfg = dataset.cfg
            model.config(cfg)
            for i, (feature_robot, label, feature_task, feature_obstacle) in enumerate(dataloader):
                feature_robot, label, feature_task, feature_obstacle = feature_robot.to(DEVICE), label.to(DEVICE), feature_task.to(DEVICE), feature_obstacle.to(DEVICE)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(feature_robot, feature_task, feature_obstacle, is_train=True)
                loss = criterion(outputs.reshape(-1, 1+cfg["n_task"]), label.reshape(-1).long())

                loss_detached = loss.detach().item()
                running_loss += loss_detached

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Print statistics
                if i == len(dataloader) - 1:
                    loss_ave = running_loss / len(dataloader)
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Scale [{i_scale + 1}/{n_scale}], Loss: {loss_ave:.4f}')
                    logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Scale [{i_scale + 1}/{n_scale}], Loss: {loss_ave:.4f}')
                    running_loss = 0.0
                    
        # test
        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            right = 0
            error = 0
            count = 0
            for i_scale in range(n_scale_test):
                dataset = datasets_test[i_scale]
                dataloader = dataloaders_test[i_scale]
                cfg = dataset.cfg
                model.config(cfg)
                for i, (feature_robot, label, feature_task, feature_obstacle) in enumerate(dataloader):
                    feature_robot, label, feature_task, feature_obstacle = feature_robot.to(DEVICE), label.to(DEVICE), feature_task.to(DEVICE), feature_obstacle.to(DEVICE)
                    outputs = model(feature_robot, feature_task, feature_obstacle, is_train=False)
                    model_label = torch.argmax(outputs, dim=-1)
                    #如果model_label和label相等，则right += 1,否则error += 1
                    right += torch.sum(model_label == label).item()
                    error += torch.sum(model_label != label).item()
                    loss = criterion(outputs.reshape(-1, 1+cfg["n_task"]), label.reshape(-1).long())
                    test_loss += loss.detach().item()
                    count += 1
            
            acc_ave = 100.0 * right / (right + error)
            test_loss_ave = test_loss / count
            print(f'Epoch [{epoch + 1}/{num_epochs}], Test Loss: {test_loss_ave:.4f}, Test Accuracy: {acc_ave:.4f}%')
            logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Test Loss: {test_loss_ave:.4f}, Test Accuracy: {acc_ave:.4f}%')
                    
            if acc_ave > max_acc:
                max_acc = acc_ave
                torch.save(model.state_dict(), os.path.join(save_dir, "time_{}_acc_{:.2f}.pt".format
                                                (time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()), acc_ave)))
                print('Model saved')
                logging.info('Model saved')
                

    print('Finished Training')
    logging.info('Finished Training')

if __name__ == "__main__":
    logging.basicConfig(filename='/home/users/wzr/project/no_com_sim/log/train_intention_{}.log'.format(TM.strftime("%Y-%m-%d-%H-%M", TM.localtime())),
                         level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logging.info('BEGIN')
    train_intention_net()