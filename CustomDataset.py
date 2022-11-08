from torch.utils.data import Dataset
import random 
import os
import torch

def GLOBAL_to_LGR_path(global_lists, key, names, var):
    lgr_list = []
    for path in global_lists:
        case = path.split('/')[-1]
        slope = case[:7]
        idx = case.split('_')[2]
        for nwell in range(1,5):
            if var == 'dP':
                string = f'{slope}_{idx}_{key}_WELL{nwell}_DP.pt'
                if string in names:
                    home_path = f'/dP_{key}/'
                    lgr_list.append(home_path + string)
            elif var == 'SG':
                string = f'{slope}_{idx}_{key}_WELL{nwell}_SG.pt'
                if string in names:
                    home_path = f'/SG_{key}/'
                    lgr_list.append(home_path + string)
                    
    return lgr_list

class CustomDataset(Dataset):
    def __init__(self, root_path, names):
        self.names = names
        self.root_path = root_path
        
    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        path = self.names[idx]
        data = torch.load(self.root_path+path)
        
        name = path.split('/')[-1]
        slope, idx, well = name[:7], name.split('_')[2], name.split('_')[-2]
        
        x = data['input'].permute(0,4,1,2,3,5)[0,...]
        y = data['output'].permute(0,4,1,2,3,5)[0,...,:1]
    
        D = {'x': x, 
             'y': y,
             'path': [slope, idx, well]}
        return D