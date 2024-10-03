import torch
import os
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from intention_judgment import IntentionNet
from dataset_intention import IntentionDataset

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
    robot_n = 5
    task_n = 5
    batch_size = 128
    attention_head = 8
    num_epochs = 50
    learning_rate = 0.001

    # Initialize model, loss function, and optimizer
    model = IntentionNet(embedding_size, robot_n, task_n, batch_size, attention_head).to(DEVICE)
    save_dir = '/home/users/wzr/project/no_com_sim/model_save/intention/'
    min = 0.1
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load dataset
    map_dirs = "/home/users/wzr/project/no_com_sim/intention_data/"
    n_map = 100
    dataset = IntentionDataset(map_dirs, n_map, robot_n, task_n)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (feature_robot, label, feature_task) in enumerate(dataloader):
            feature_robot, label, feature_task = feature_robot.to(DEVICE), label.to(DEVICE), feature_task.to(DEVICE)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(feature_robot, feature_task, is_train=True)
            loss = criterion(outputs.view(-1, task_n + 1), label.view(-1).long())

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # Print every 10 mini-batches
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {running_loss / 10:.4f}')
                if running_loss < min:
                    min = running_loss
                    torch.save(model.state_dict(), os.path.join(save_dir,'date{}-epoch{}-i{}-coss_{:.5f}.pt'.format(
                        time.strftime("%Y-%m-%d", time.localtime()), epoch, i, running_loss)))
                running_loss = 0.0

    print('Finished Training')

if __name__ == "__main__":
    train_intention_net()