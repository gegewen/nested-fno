import json
from torch.utils.data import Dataset
import random 
import os
import torch
import glob
from CustomDataset import *
random.seed(0)

files = os.listdir('ECLIPSE/meta_data/')
names = []
for file in files:
    n = file.split('.')[0]
    names.append(f'{n}_GLOBAL_DP.pt')
    
random.shuffle(names)
train_names = names[:16]
val_names = names[16:18]
test_names = names[18:]

ROOT_PATH = 'dataset/'
DATA_LOADER_DICT = {}
DATA_LOADER_DICT['GLOBAL'] = {'train': CustomDataset(ROOT_PATH+'dP_GLOBAL/', train_names),
                              'val': CustomDataset(ROOT_PATH+'dP_GLOBAL/', val_names),
                              'test':  CustomDataset(ROOT_PATH+'dP_GLOBAL/', test_names)}

for key in ['LGR1', 'LGR2', 'LGR3', 'LGR4']:
    LGR_ROOT_PATH = os.listdir(f'{ROOT_PATH}dP_{key}/')
    train_lgr_lists_dP = GLOBAL_to_LGR_path(train_names, key, LGR_ROOT_PATH, 'dP')
    val_lgr_lists_dP = GLOBAL_to_LGR_path(val_names, key, LGR_ROOT_PATH, 'dP')
    test_lgr_lists_dP = GLOBAL_to_LGR_path(test_names, key, LGR_ROOT_PATH, 'dP')
    LGR_ROOT_PATH = os.listdir(f'{ROOT_PATH}SG_{key}/')
    train_lgr_lists_SG = GLOBAL_to_LGR_path(train_names, key, LGR_ROOT_PATH, 'SG')
    val_lgr_lists_SG = GLOBAL_to_LGR_path(val_names, key, LGR_ROOT_PATH, 'SG')
    test_lgr_lists_SG = GLOBAL_to_LGR_path(test_names, key, LGR_ROOT_PATH, 'SG')
    
    DATA_LOADER_DICT[key] = {'dP': {'train': CustomDataset(ROOT_PATH, train_lgr_lists_dP), 
                        'val': CustomDataset(ROOT_PATH, val_lgr_lists_dP), 
                        'test': CustomDataset(ROOT_PATH, test_lgr_lists_dP)},
                           'SG': {'train': CustomDataset(ROOT_PATH, train_lgr_lists_SG), 
                        'val': CustomDataset(ROOT_PATH, val_lgr_lists_SG), 
                        'test': CustomDataset(ROOT_PATH, test_lgr_lists_SG)}}
    
torch.save(DATA_LOADER_DICT, 'DATA_LOADER_DICT.pth')
print('data loader done')