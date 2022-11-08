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

META_PATH = '..'
meta_files = glob.glob(f'{META_PATH}/ECLIPSE/meta_data/*.npy')
print('meta data:', len(meta_files))

PT_PATH = f'../dataset/dP_GLOBAL/'
if not os.path.exists(PT_PATH):
    os.mkdir(PT_PATH)

pt_files = os.listdir(PT_PATH)
print('done collected:', len(pt_files))

collect_index = []
for name in meta_files:
    name = name.split('/')[-1][:-4]
    if f'{name}_GLOBAL_DP.pt' not in pt_files:
        collect_index.append([name[:7], int(name.split('_')[2])])
print('to collect', len(collect_index))

NX, NY, NZ, NT = 100, 100, 5, 24

for tup in collect_index:
    case_path, idx = tup
    case_name = f'case_{idx}'
    meta_data = np.load(f'{META_PATH}/ECLIPSE/meta_data/{case_path}_{idx}.npy', 
                        allow_pickle=True).tolist()

    for k, v in meta_data[case_name].items():
        globals()[k]=v

    OUTPUT_DICT = return_OUTPUT_DICT(meta_data, case_name)
    
    p, t, rate = INPUT_DICT['p'], INPUT_DICT['temp'], INPUT_DICT['inj']
    INJ_MAP_DICT = return_inj_map_dict(WELL_LIST,rate,INJ_LOCATION_DICT,GRID_CENTER_DICT,LGR_LIST)

    gridx = np.repeat(xy_norm(GRID_CENTER_DICT['GLOBAL']['grid_x'])[...,None,None], 24, axis=-2)
    gridy = np.repeat(xy_norm(GRID_CENTER_DICT['GLOBAL']['grid_y'])[...,None,None], 24, axis=-2)
    gridz = np.repeat(z_norm(TOPS_DICT['GLOBAL'][0,...,None,None]), 24, axis=-2)
    gridt = (np.ones(gridz.shape)* times[None,None,None,:,None])

    inj = np.repeat(INJ_MAP_DICT['GLOBAL'][...,None,None], 24, axis=-2)
    pressure = np.repeat(p_norm(return_upsample_dict(OUTPUT_DICT, 0, 'BPR', 
                                           WELL_LIST, GRID_IDX_DICT)['GLOBAL'][...,None,None]), 
                         24, axis=-2)
    temp = t_norm(t) * np.ones(inj.shape)
    perm = np.repeat(k_norm(PERM_DICT['GLOBAL'])[0,...,None,None], 24, axis=-2)

    DICT = return_upsample_all_time(OUTPUT_DICT, 'dP', WELL_LIST, GRID_IDX_DICT, LGR_LIST)

    x_DP = np.concatenate([gridx, gridy, gridz, gridt, inj, pressure, temp, perm], 
                          axis=-1)[None,...]
    y_DP = DICT['GLOBAL'][...,None]

    x_DP = torch.from_numpy(x_DP.astype(np.float32))
    y_DP = torch.from_numpy(y_DP.astype(np.float32))

    data = {}
    data['input'] = x_DP
    data['output'] = y_DP
    
    torch.save(data, f'../dataset/dP_GLOBAL/{case_path}_{idx}_GLOBAL_DP.pt')
    print(f'{case_path}_{idx}_GLOBAL_DP done')