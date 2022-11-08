from torch.utils.data import Dataset
import random 
import os
import torch
from CustomDataset import *


DATA_LOADER_DICT = torch.load('DATA_LOADER_DICT.pth')
train_loader = DATA_LOADER_DICT['GLOBAL']['train']
val_loader = DATA_LOADER_DICT['GLOBAL']['val']
n_train = len(train_loader)
n_val = len(val_loader)
print(n_train, n_val)



from lploss import *
LPloss = LpLoss(size_average=True)

from FNO4D import *
device = torch.device('cuda')
width = 28
mode1, mode2, mode3, mode4 = 4, 20, 20, 2
model_global = FNO4d(mode1, mode2, mode3, mode4, width, in_dim=8)
model_global.to(device)


from datetime import datetime
from datetime import date


now = datetime.now()
today = date.today()

day = today.strftime("%m%d")
current_time = now.strftime("%H%M")
specs = f'FNO4D-GLOBAL-DP'
model_str = f'{day}-{current_time}-train{n_train}'
print(f'{specs}-{model_str}')


from Adam import Adam
scheduler_step = 5
scheduler_gamma = 0.85
learning_rate = 1e-3

optimizer = Adam(model_global.parameters(), lr=learning_rate, weight_decay=0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                        step_size=scheduler_step, 
                                        gamma=scheduler_gamma)


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(f'logs/')


import pickle
from UnitGaussianNormalizer import *
    
with open("normalizer/input_normalizer_GLOBAL_DP_val.pickle", 'rb') as f:
    input_normalizer_GLOBAL = pickle.load(f)
    input_normalizer_GLOBAL.cuda()
    
with open("normalizer/output_normalizer_GLOBAL_DP_val.pickle", 'rb') as f:
    output_normalizer_GLOBAL = pickle.load(f)
    output_normalizer_GLOBAL.cuda()
    
    
for ep in range(51):
    model_global.train()
    train_lp = 0
    c = 0
    
    for data in train_loader:
        x, y, path = data['x'], data['y'], data['path']
        slope, idx = path[0], path[1]
        path = f'{slope}_{idx}'
        
        x, y = x[None,...].to(device), y[None,...].to(device)
        optimizer.zero_grad()
        
        x[...,-1:] = input_normalizer_GLOBAL.encode(x[...,-1:])
        pred = model_global(x)
        pred = output_normalizer_GLOBAL.decode(pred)
        loss = LPloss(pred.reshape(1, -1), y[...,:1].reshape(1, -1))
        train_lp += loss.item()
        
        loss.backward()
        optimizer.step()
        c += 1
        if c%10 ==0:
            writer.add_scalars('dP LPloss', {f'{model_str}_{specs}_train': loss.item()}, 
                               ep*n_train+c)            
            print(f'ep: {ep}, iter: {c}, train lp: {loss.item():.4f}')       

    scheduler.step()
    
    model_global.eval()
    val_lp = 0
    val_mre = 0
    with torch.no_grad():
        for data in val_loader:
            x, y, path = data['x'], data['y'], data['path']
            slope, idx = path[0], path[1]
            path = f'{slope}_{idx}'
            
            x, y = x[None,...].to(device), y[None,...].to(device)
            x[...,-1:] = input_normalizer_GLOBAL.encode(x[...,-1:])
            pred = model_global(x)
            pred = output_normalizer_GLOBAL.decode(pred)
            loss = LPloss(pred.reshape(1, -1), y[...,:1].reshape(1, -1))
            val_lp += loss.item()
    
    writer.add_scalars('dP LPloss', {f'{model_str}_{specs}_train': train_lp/n_train, 
                               f'{model_str}_{specs}_val': val_lp/n_val}, ep*n_train+c)
    
    print(f'epoch: {ep} summary')
    print(f'train loss: {train_lp/n_train:.4f}, val loss: {val_lp/n_val:.4f}')
    print('----------------------------------------------------------------------')

    torch.save(model_global, f'saved_models/{model_str}-{specs}-ep{ep}.pt')