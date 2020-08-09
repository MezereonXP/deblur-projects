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

from PIL import Image
import torchvision.transforms as transforms

my_transform = transforms.Compose([ 
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (1,1,1)),
                ])

def run():
    # Load the config file
    f = open('./config.yml')
    config = edict(yaml.load(f))
    print("The config:")
    print(config)
    print("-------------------------------------------------------------------")

    device = config.device
    mode = config.mode
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
    
    # Testing setting
    if mode == 'test':
        model = model.eval()
        run_test(config, model)
        exit

    # Optimization Setting
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay_rate)
    if optim_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=decay_rate)
    
    # Loss Setting
    mse_loss = torch.nn.MSELoss(reduction = 'sum')

    # Tensorboard Setting
    writer = SummaryWriter('./tensorboard_history')

    # Load the dataset
    dataset = GoProDataset(path=config.data_path,mode=config.mode)
    mydataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    epoches = int(config.epoch)

    i = 0
    for epoch in range(epoches):
        training_loss = 0
        for batch_x, batch_y in mydataloader:
            i += 1
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
            # writer.add_image('output', bo, global_step=i) 
            # writer.add_scalar('loss', loss.data.item(), i)
            print('Epoch:{}|num:{}|loss:{}'.format(epoch, i, loss.data.item()))
    
        # Write the scalar
        bx = batch_x[0].unsqueeze(0)
        bo = batch_out[0].unsqueeze(0)
        by = batch_y[0].unsqueeze(0)
        # print(bo)
        grid_data = torch.cat((torch.cat((bx,by),dim=0),bo),dim=0)
        img_grid = vutils.make_grid(grid_data, normalize=True)
        writer.add_image('image', img_grid, global_step=epoch)
        writer.add_scalar('global loss', training_loss/len(dataset), epoch)
        if (epoch+1)%100 == 0:
            print("Saving model......")
            torch.save(model, config.save_path+"/"+config.save_name)

def divide(input_img):
    block = []
    height = input_img.shape[1]
    width = input_img.shape[2]
    flag_h = 0 if height%256 == 0 else 1
    flag_w = 0 if width%256 == 0 else 1
    for i in range(int(height/256)+flag_h):
        for j in range(int(width/256)+flag_w):
            h_start = i*256
            h_end = (i+1)*256
            w_start = j*256
            w_end = (j+1)*256
            if h_end > height:
                h_end = height
                h_start = -256
            if w_end > width:
                w_end = width
                w_start = -256
            if len(block) == 0:
                block = input_img[:][h_start:h_end][w_start:w_end].unsqueeze(0)
            else:
                tmp = input_img[:][h_start:h_end][w_start:w_end].unsqueeze(0)
                print(tmp.shape)
                print(block.shape)
                block = torch.cat((block, tmp), dim=0)
    return block, width, height

def combine(imgs, w, h):
    result = torch.zeros(3, h, w)
    flag_h = 0 if h%256 == 0 else 1
    flag_w = 0 if w%256 == 0 else 1
    for i in range(int(h/256)+flag_h):
        for j in range(int(w/256)+flag_w):
            h_start = i*256
            h_end = (i+1)*256
            w_start = j*256
            w_end = (j+1)*256
            if h_end > height:
                h_end = height
                h_start = -256
            if w_end > width:
                w_end = width
                w_start = -256
            
            result[:][h_start:h_end][w_start:w_end] = imgs[i*(int(w/256)+flag_h)+j][:][:][:]
    return result

def run_test(config, model):
    test_img_path = config.test_img_path
    output_path = config.output_path
    output_name = config.output_name
    
    input_img = Image.open(test_img_path).convert('RGB')
    input_img = my_transform(input_img)
    # Divide into multi 256x256 blocks
    blocks, width, height = divide(input_img)
    outputs = model(blocks)
    output_img = combine(outputs, width, height)
    vutils.save_image(output_img, output_path+'/'+output_name, normalize=True)
    print('Saved Result in {}'.format(output_path))



if __name__ == "__main__":
    run()