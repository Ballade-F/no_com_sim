import torch
import os
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from intention import IntentionNet
from dataset_intention import IntentionDataset
import json

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
    dataset_dir = "/home/data/wzr/no_com_1/data/intention"
    # read json file
    with open(os.path.join(dataset_dir, "dataset_info.json"), "r") as f:
        dataset_info = json.load(f)
    n_scale = dataset_info["n_scale"]
    ob_points = dataset_info["ob_points"]

    # Initialize model, loss function, and optimizer
    model = IntentionNet(ob_points, r_points, embedding_size, batch_size, attention_head, device=DEVICE).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    datasets = [IntentionDataset(os.path.join(dataset_dir, f"scale_{i}"), r_points) for i in range(n_scale)]
    dataloaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True) for dataset in datasets]

    min_loss = 0.1
    running_loss = 0.0
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

                running_loss += loss.item()

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Print statistics
                if i == len(dataloader) - 1:
                    loss_ave = running_loss / len(dataloader)
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Scale [{i_scale + 1}/{n_scale}], Loss: {loss_ave:.4f}')
                    running_loss = 0.0
                    if loss_ave < min_loss:
                        min_loss = loss_ave
                        torch.save(model.state_dict(), os.path.join(save_dir, "time_{}_loss_{:.2f}.pt".format
                                                      (time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()), min_loss)))
                        print('Model saved')
                

    print('Finished Training')

if __name__ == "__main__":
    train_intention_net()