#!/usr/bin/python
import sys
from torch.utils.data import Dataset
import random 
import os
import torch
from CustomDataset import *

key = sys.argv[1]
var = 'dP'

DATA_LOADER_DICT = torch.load('DATA_LOADER_DICT.pth')
train_loader = DATA_LOADER_DICT[key][var]['train']
val_loader = DATA_LOADER_DICT[key][var]['val']
n_train = len(train_loader)
n_val = len(val_loader)
print(n_train, n_val)

from lploss import *
LPloss = LpLoss(size_average=True)

from FNO4D import *
device = torch.device('cuda')
width = 28
mode1, mode2, mode3, mode4 = 6, 10, 10, 10
model = FNO4d(mode1, mode2, mode3, mode4, width, in_dim=9)
model.to(device)

from datetime import datetime
from datetime import date

now = datetime.now()
today = date.today()

day = today.strftime("%m%d")
current_time = now.strftime("%H%M")
specs = f'FNO4D-{key}-{var}'
model_str = f'{day}-{current_time}-train{n_train}'
print(f'{specs}-{model_str}')


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(f'logs/')


import pickle
from UnitGaussianNormalizer import *
    
with open(f"normalizer/input_normalizer_{key}_{var.upper()}_val.pickle", 'rb') as f:
    input_normalizer = pickle.load(f)
    input_normalizer.cuda()
    
with open(f"normalizer/output_normalizer_{key}_{var.upper()}_val.pickle", 'rb') as f:
    output_normalizer = pickle.load(f)
    output_normalizer.cuda()
    
    
from Adam import Adam
scheduler_step = 5
scheduler_gamma = 0.85
learning_rate = 1e-3

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                        step_size=scheduler_step, 
                                        gamma=scheduler_gamma)

for ep in range(11):
    model.train()
    train_lp = 0
    c = 0
    
    for data in train_loader:
        x, y, path = data['x'], data['y'], data['path']
        slope, idx = path[0], path[1]
        path = f'{slope}_{idx}'
        
        x, y = x[None,...].to(device), y[None,...][...,:1].to(device)
        
        optimizer.zero_grad()
        x[...,-1:] = input_normalizer.encode(x[...,-1:])
        pred = model(x)
        pred = output_normalizer.decode(pred)        
        loss = LPloss(pred.reshape(1, -1), y.reshape(1, -1))        
        train_lp += loss.item()
        
        loss.backward()
        optimizer.step()
        c += 1
        
        if c%10 ==0:
            writer.add_scalars(f'{var} LPloss', {f'{model_str}_{specs}_train': loss.item()}, 
                               ep*n_train+c)
            print(f'ep: {ep}, iter: {c}, train lp: {loss.item():.4f}')

    scheduler.step()
    
    model.eval()
    val_lp = 0
    val_mre = 0
    with torch.no_grad():
        for data in val_loader:
            x, y, path = data['x'], data['y'], data['path']
            slope, idx = path[0], path[1]
            path = f'{slope}_{idx}'
            
            x, y = x[None,...].to(device), y[None,...][...,:1].to(device)
            x[...,-1:] = input_normalizer.encode(x[...,-1:])
            pred = model(x)
            pred = output_normalizer.decode(pred)
            loss = LPloss(pred.reshape(1, -1), y.reshape(1, -1))
            val_lp += loss.item()
    
    writer.add_scalars(f'{var} LPloss', {f'{model_str}_{specs}_train': train_lp/n_train, 
                               f'{model_str}_{specs}_val': val_lp/n_val}, ep*n_train+c)

    print(f'epoch: {ep} summary')
    print(f'train loss: {train_lp/n_train:.4f}, val loss: {val_lp/n_val:.4f}')
    print('----------------------------------------------------------------------')

    torch.save(model, f'saved_models/{model_str}-{specs}-ep{ep}.pt')