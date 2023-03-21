from iapws import IAPWS97
import torch
import numpy as np
# from config_pt_utility import *


xy_norm = lambda x: (x)/160000
z_norm = lambda x: (x-2000)/2000
p_norm = lambda x: (x)/172
t_norm = lambda x: (x)/70
k_norm = lambda x: (x)/100
z_dnorm = lambda x: x*2000+2000
p_dnorm = lambda x: 172*x
depth_func = lambda a : (a - 1.01325)/9.8*100

times = np.cumsum(10*np.array(np.power(1.2531,np.arange(1,25,1)), dtype=int))
times = times/ 10950

def calculate_p_initial(DEPTH_datum, P_datum, T_datum, tops, thickness):
    rho_datum = IAPWS97(T=T_datum+273.15, P=P_datum/10).Liquid.rho
    Z_cell_center = tops+thickness/2
    return  P_datum + (Z_cell_center-DEPTH_datum)* 9.8*rho_datum/100000

def create_P_INIT_DICT(TOPS_DICT, INPUT_DICT, WELL_LIST, LGR_LIST, GRID_IDX_DICT):
    DEPTH_datum = depth_func(INPUT_DICT['p'])
    P_datum = INPUT_DICT['p']
    T_datum = INPUT_DICT['temp']

    P_INIT_DICT = {}
    P_INIT_DICT['GLOBAL'] = calculate_p_initial(DEPTH_datum, P_datum, T_datum, 
                             tops=TOPS_DICT['GLOBAL'], 
                             thickness=GRID_IDX_DICT['GLOBAL']['DZ'])
    for well in WELL_LIST:
        d = {}
        for lgr in LGR_LIST:
            d[lgr] = calculate_p_initial(DEPTH_datum, P_datum, T_datum, 
                             tops=TOPS_DICT[well][lgr], 
                             thickness=GRID_IDX_DICT[well][lgr]['DZ'])
        P_INIT_DICT[well] = d
    return P_INIT_DICT

def meta_data_to_input_dict(meta_data):
    for k, v in meta_data.items():
        globals()[k]=v

    p, t, rate = INPUT_DICT['p'], INPUT_DICT['temp'], INPUT_DICT['inj']
    P_INIT_DICT = create_P_INIT_DICT(TOPS_DICT, INPUT_DICT, WELL_LIST, LGR_LIST, GRID_IDX_DICT)
    INJ_MAP_DICT = return_inj_map_dict(WELL_LIST,rate,INJ_LOCATION_DICT,GRID_CENTER_DICT,LGR_LIST)
    pressure_upsampled_dict = return_upsample_dict(P_INIT_DICT, WELL_LIST, GRID_IDX_DICT)

    ml_input_dict = {}

    # GLOBAL
    gridx = np.repeat(xy_norm(GRID_CENTER_DICT['GLOBAL']['grid_x'])[...,None,None], 24, axis=-2)
    gridy = np.repeat(xy_norm(GRID_CENTER_DICT['GLOBAL']['grid_y'])[...,None,None], 24, axis=-2)
    gridz = np.repeat(z_norm(TOPS_DICT['GLOBAL'][0,...,None,None]), 24, axis=-2)
    gridt = (np.ones(gridz.shape)* times[None,None,None,:,None])
    inj = np.repeat(INJ_MAP_DICT['GLOBAL'][...,None,None], 24, axis=-2)
    temp = t_norm(t) * np.ones(inj.shape)
    perm = np.repeat(k_norm(PERM_DICT['GLOBAL'])[0,...,None,None], 24, axis=-2)
    pressure = np.repeat(p_norm(pressure_upsampled_dict['GLOBAL'][...,None,None]), 24, axis=-2)
    x = np.concatenate([gridx, gridy, gridz, gridt, inj, pressure, temp, perm], axis=-1)[None,...].transpose(0,4,1,2,3,5)
    x = torch.from_numpy(x.astype(np.float32))
    ml_input_dict['GLOBAL'] = x

    for well in WELL_LIST:
        well_input_dict = {}
        # LGR1
        gridx = np.repeat(xy_norm(GRID_CENTER_DICT[well]['LGR1']['grid_x'])[...,None,None], 24, axis=-2)
        gridy = np.repeat(xy_norm(GRID_CENTER_DICT[well]['LGR1']['grid_y'])[...,None,None], 24, axis=-2)
        gridz = np.repeat(z_norm(TOPS_DICT[well]['LGR1'][0,...,None,None]), 24, axis=-2)
        gridt = (np.ones(gridz.shape)* times[None,None,None,:,None])
        inj = np.repeat(INJ_MAP_DICT[well]['LGR1'][...,None,None], 24, axis=-2)
        pressure = np.repeat(p_norm(pressure_upsampled_dict[well]['LGR1'][0,...,None,None]), 24, axis=-2)
        temp = t_norm(t) * np.ones(inj.shape)
        perm = np.repeat(k_norm(PERM_DICT[well]['LGR1'])[0,...,None,None], 24, axis=-2)
        I1, I2 = GRID_IDX_DICT[well]['LGR1']['I1']-1-15, GRID_IDX_DICT[well]['LGR1']['I2']+15
        J1, J2 = GRID_IDX_DICT[well]['LGR1']['J1']-1-15, GRID_IDX_DICT[well]['LGR1']['J2']+15
        coarse = np.zeros(gridx.shape)
        x = np.concatenate([gridx, gridy, gridz, gridt, inj, pressure, temp, perm, coarse], axis=-1)[None,...]
        x = torch.from_numpy(x.astype(np.float32))
        well_input_dict['LGR1'] = x

        # LGR2-4
        for lgr in ['LGR2', 'LGR3', 'LGR4']:
            gridx = np.repeat(xy_norm(GRID_CENTER_DICT[well][lgr]['grid_x'])[...,None,None], 24, axis=-2)
            gridy = np.repeat(xy_norm(GRID_CENTER_DICT[well][lgr]['grid_y'])[...,None,None], 24, axis=-2)
            gridz = np.repeat(z_norm(TOPS_DICT[well][lgr][0,...,None,None]), 24, axis=-2)
            gridt = (np.ones(gridz.shape)* times[None,None,None,:,None])
            inj = np.repeat(INJ_MAP_DICT[well][lgr][...,None,None], 24, axis=-2)
            pressure = np.repeat(p_norm(pressure_upsampled_dict[well][lgr][0,...,None,None]), 24, axis=-2)
            temp = t_norm(t) * np.ones(inj.shape)
            perm = np.repeat(k_norm(PERM_DICT[well][lgr])[0,...,None,None], 24, axis=-2)
            x = np.concatenate([gridx, gridy, gridz, gridt, inj, pressure, temp, perm, np.zeros(gridx.shape)], axis=-1)[None,...]
            x = torch.from_numpy(x.astype(np.float32))
            if lgr == 'LGR4':
                perf = PERF_DICT[well]
                old_inj = torch.clone(x[0,19,19,:,:,4])
                new_inj = torch.zeros(old_inj.shape)
                new_inj[perf[0]-1:perf[1],:] = old_inj[perf[0]-1:perf[1],:]
                x[0,19,19,:,:,4] = new_inj
            well_input_dict[lgr] = x
        ml_input_dict[well] = well_input_dict
    return ml_input_dict