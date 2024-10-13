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
    batch_size = 16
    attention_head = 8
    r_points = 5
    num_epochs = 10
    learning_rate = 0.001

    save_dir = '/home/ballade/Desktop/Project/no_com_sim/net_model/intention/'
    dataset_dir = "/home/ballade/Desktop/Project/no_com_sim/intention_data/"
    # read json file
    with open(os.path.join(dataset_dir, "dataset_info.json"), "r") as f:
        dataset_info = json.load(f)
    n_scale = dataset_info["n_scale"]
    ob_points = dataset_info["ob_points"]

    # Initialize model, loss function, and optimizer
    model = IntentionNet(ob_points, r_points, embedding_size, batch_size, attention_head, device=DEVICE).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    running_loss = 0.0
    counter = 0
    # epoch training
    for epoch in range(num_epochs):
        model.train()
        for i_scale in range(n_scale):
            dataset = IntentionDataset(os.path.join(dataset_dir, f"scale_{i_scale}"), r_points)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
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
                counter += 1

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

        # Print statistics
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / counter:.4f}')
                

    print('Finished Training')

if __name__ == "__main__":
    train_intention_net()