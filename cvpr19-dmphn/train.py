import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils

import yaml
import os
from easydict import EasyDict as edict
from dataset import GoProDataset
from torch.utils.data.dataloader import DataLoader
from model import DMPHNModel
from tensorboardX import SummaryWriter


def run():
    # Load the config file
    f = open('./config.yml')
    config = edict(yaml.load(f))
    print("The config:")
    print(config)
    print("-------------------------------------------------------------------")

    device = config.device
    lr = float(config.lr)
    decay_rate = float(config.decay_rate)
    batch_size = int(config.batch_size)
    optim_type = config.solver
    save_path_with_name = config.save_path+"/"+config.save_name
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)

    # Load Model
    model = DMPHNModel(level=4, device=device).to(device)

    # History reload
    if not config.model_path == 'none':
        model = torch.load(config.model_path)

    # Optimization Setting
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay_rate)
    if optim_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=decay_rate)
    
    # Loss Setting
    mse_loss = torch.nn.MSELoss()

    # Tensorboard Setting
    writer = SummaryWriter('./tensorboard_history')

    # Load the dataset
    dataset = GoProDataset(path=config.data_path,mode=config.mode)
    mydataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    epoches = int(config.epoch)

    for epoch in range(epoches):
        training_loss = 0
        for batch_x, batch_y in mydataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # forward
            # print(batch_x.shape)
            batch_out = model(batch_x)
            loss = mse_loss(batch_out, batch_y)
            training_loss += loss.data.item()
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Write the scalar
        writer.add_scalar('loss', training_loss/len(dataset), epoch)
        bx = batch_x[0].unsqueeze(0)
        bo = batch_out[0] + 0.5  # Un-normalization
        by = batch_y[0].unsqueeze(0)
        print(bo)
        grid_data = torch.cat((bx,by),dim=0)
        img_grid = vutils.make_grid(grid_data, normalize=True)
        writer.add_image('input', img_grid, global_step=epoch)
        writer.add_image('output', bo, global_step=epoch) 
        print('Epoch:{}|loss:{}'.format(epoch, training_loss/len(dataset)))
        if (epoch+1)%100 == 0:
            print("Saving model......")
            torch.save(model, config.save_path+"/"+config.save_name)

if __name__ == "__main__":
    run()