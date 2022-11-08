import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
import glob
import os
from config_utility import *

import torch.nn.functional as F
import torch

xy_norm = lambda x: (x)/160000
z_norm = lambda x: (x-2000)/2000
p_norm = lambda x: (x)/172
t_norm = lambda x: (x)/70
k_norm = lambda x: (x)/100

times = np.cumsum(10*np.array(np.power(1.2531,np.arange(1,25,1)), dtype=int))
times = times/ 10950

PT_GLOBAL_PATH = f'../dataset/dP_GLOBAL/'
pt_files = os.listdir(PT_GLOBAL_PATH)
print('done collected:', len(pt_files))

GLOBAL_names = []
for file in pt_files:
    l = file.split('_')
    GLOBAL_names.append((f'{l[0]}_{l[1]}', int(l[2])))
print(len(GLOBAL_names))

# find reservoirs that has not been collected
path = f'../dataset/SG_LGR3/'
if not os.path.exists(path):
    os.mkdir(path)
files = os.listdir(path)


collected_names = []
for file in files:
    l = file.split('_')
    collected_names.append((f'{l[0]}_{l[1]}', int(l[2])))
print(len(collected_names))

to_load_names = []
for elem in GLOBAL_names:
    if elem not in collected_names:
        to_load_names.append(elem)
        
print(len(to_load_names))

NX, NY, NZ, NT = 40, 40, 50, 24
ROOT_PATH = '..'

for names in to_load_names:
    slope_name, idx = names
    case_name = f'case_{idx}'
    meta_data = np.load(f'{ROOT_PATH}/ECLIPSE/meta_data/{slope_name}_{idx}.npy', allow_pickle=True).tolist()
    
    for k, v in meta_data[case_name].items():
        globals()[k]=v
    
    OUTPUT_DICT = return_OUTPUT_DICT(meta_data, case_name)

    p, t, rate = INPUT_DICT['p'], INPUT_DICT['temp'], INPUT_DICT['inj']
    INJ_MAP_DICT = return_inj_map_dict(WELL_LIST,rate,INJ_LOCATION_DICT,GRID_CENTER_DICT, LGR_LIST)
    print(idx)
    
    for well in WELL_LIST:
        gridx = np.repeat(xy_norm(GRID_CENTER_DICT[well]['LGR3']['grid_x'])[...,None,None], 24, axis=-2)
        gridy = np.repeat(xy_norm(GRID_CENTER_DICT[well]['LGR3']['grid_y'])[...,None,None], 24, axis=-2)
        gridz = np.repeat(z_norm(TOPS_DICT[well]['LGR3'][0,...,None,None]), 24, axis=-2)
        gridt = (np.ones(gridz.shape)* times[None,None,None,:,None])
        
        inj = np.repeat(INJ_MAP_DICT[well]['LGR3'][...,None,None], 24, axis=-2)
        pressure = np.repeat(p_norm(return_upsample_dict(OUTPUT_DICT, 0, 'BPR', 
                                               WELL_LIST, GRID_IDX_DICT)[well]['LGR3'][0,...,None,None]), 24, axis=-2)
        temp = t_norm(t) * np.ones(inj.shape)
        perm = np.repeat(k_norm(PERM_DICT[well]['LGR3'])[0,...,None,None], 24, axis=-2)

        
        DICT = return_upsample_all_time(OUTPUT_DICT, 'BGSAT', WELL_LIST, 
                                        GRID_IDX_DICT, LGR_LIST)
        
        coarse = DICT[well]['LGR2'][0,:,:,:,:,None]
        x_DP = np.concatenate([gridx, gridy, gridz, gridt, inj, pressure, temp, perm, coarse], axis=-1)[None,...]
        y_DP = DICT[well]['LGR3'][...,None]

        x_DP = torch.from_numpy(x_DP.astype(np.float32))
        y_DP = torch.from_numpy(y_DP.astype(np.float32))

        data = {}
        data['input'] = x_DP
        data['output'] = y_DP
        print(f'{slope_name}_{idx}_LGR3_{well}_SG.pt')
        
        torch.save(data, f'../dataset/SG_LGR3/{slope_name}_{idx}_LGR3_{well}_SG.pt')