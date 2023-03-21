import torch.nn.functional as F
import torch 
import numpy as np
import json
import copy 
from vonk3d import *

def user_input_to_meta_data(ax_m, ay_m, az_m, mu, std, pressure, temp, dip, inj_loc, inj_rate, PERF_DICT):
    n_well = len(inj_rate)
    WELL_LIST = [f'WELL{i_well+1}' for i_well in range(n_well)]

    coord_well = []
    for well in WELL_LIST:
        coord_well.append([int(inj_loc[well][0]//1600-2), int(inj_loc[well][1]//1600)])

    kmap3d = vonk3d(rseed=0,
                    dx=1,dy=1,dz=1,
                    ax=ax_m/400,ay=ay_m/400,az=az_m/2,
                    ix=400,iy=400,iz=50,
                    pop=1,med=3,nu=1)

    kmap3d = kmap3d*std + mu
    kmap3d = np.exp(kmap3d)

    PERM_DICT = return_PERM_DICT(kmap3d, coord_well)
    GRID_IDX_DICT = grid_idx_dict(coord_well)
    GRID_CENTER_DICT = grid_dict(WELL_LIST, GRID_IDX_DICT)
    LGR_LIST = list(GRID_IDX_DICT['WELL1'].keys())
    PCOLOR_GRID_DICT = pcolor_grid_dict(WELL_LIST, GRID_IDX_DICT)
    INJ_LOCATION_DICT = inj_location_dict(PCOLOR_GRID_DICT, WELL_LIST)
    INPUT_DICT = input_dict(pressure, temp, inj_rate)
    TOPS_DICT = return_tops_dict(pressure, dip, GRID_CENTER_DICT, WELL_LIST, LGR_LIST)

    meta_data = {
                'GRID_IDX_DICT':GRID_IDX_DICT, 
                'GRID_CENTER_DICT':GRID_CENTER_DICT,
                'PCOLOR_GRID_DICT':PCOLOR_GRID_DICT,
                'LGR_LIST':LGR_LIST, 
                'WELL_LIST':WELL_LIST,
                'PERM_DICT':PERM_DICT,
                'INPUT_DICT':INPUT_DICT,
                'INJ_LOCATION_DICT': INJ_LOCATION_DICT,
                'PERF_DICT':PERF_DICT,
                'TOPS_DICT':TOPS_DICT
                }

    return meta_data


depth_func = lambda a : (a - 1.01325)/9.8*100
pressure_func = lambda a: a/100*9.8 + 1.01325

def return_tops(depth, slope, grid_x, grid_z):
    nz = grid_z.shape[-1]
    adj = np.tan(np.deg2rad(slope)) * (grid_x - 160000/2)
    return adj + depth + grid_z

def return_tops_dict(pressure, dip, GRID_CENTER_DICT, WELL_LIST, LGR_LIST):
    TOPS_DICT = {}
    TOPS_DICT['GLOBAL'] = return_tops(depth_func(pressure), dip, 
                                      GRID_CENTER_DICT['GLOBAL']['grid_x'], GRID_CENTER_DICT['GLOBAL']['grid_z'])[None,...]
    for well in WELL_LIST:
        d = {}
        for lgr in LGR_LIST:
            tops = return_tops(depth_func(pressure), dip, 
                               GRID_CENTER_DICT[well][lgr]['grid_x'], 
                               GRID_CENTER_DICT[well][lgr]['grid_z'])[None,...]
            d[lgr] = tops
        TOPS_DICT[well] = d
    return TOPS_DICT


def grid_idx_dict(coord_well):
    with open('GRID_IDX_DICT.json') as f:
        GRID_IDX_DICT = json.load(f)

    n_well = len(coord_well)
    for i_well in range(n_well-1):
        GRID_IDX_DICT[f'WELL{i_well+2}'] = copy.deepcopy(GRID_IDX_DICT['WELL1'])

    for i_well in range(n_well):
        well_x_start, well_x_end = coord_well[i_well][0] - 4, coord_well[i_well][0] + 5
        well_y_start, well_y_end = coord_well[i_well][1] - 4, coord_well[i_well][1] + 5
        GRID_IDX_DICT[f'WELL{i_well+1}']['LGR1']['I1'] = well_x_start 
        GRID_IDX_DICT[f'WELL{i_well+1}']['LGR1']['I2'] = well_x_end
        GRID_IDX_DICT[f'WELL{i_well+1}']['LGR1']['J1'] = well_y_start
        GRID_IDX_DICT[f'WELL{i_well+1}']['LGR1']['J2'] = well_y_end
    return GRID_IDX_DICT

def torch_regrid(x, size):
    return F.interpolate(torch.from_numpy(x)[None, None,...], 
                          size=size, mode='trilinear', align_corners=False)[0,0,...].numpy()

def return_PERM_DICT(kmap, coord_well):
    DICT = {}
    DICT['GLOBAL'] = torch_regrid(kmap, [100, 100, 5])
    
    for i_well in range(len(coord_well)):
        d = {}
        I1, I2 = (coord_well[i_well][0] - 5)*4, (coord_well[i_well][0] + 5)*4
        J1, J2 = (coord_well[i_well][1] - 5)*4, (coord_well[i_well][1] + 5)*4
        k_LGR1 = kmap[I1:I2, J1:J2, :]
        d['LGR1'] = torch_regrid(k_LGR1, [40, 40, 25])

        I1, I2 = 18, 38
        J1, J2 = 10, 30
        d['LGR2'] = torch_regrid(k_LGR1[I1:I2, J1:J2, :], [40, 40, 50])

        I1, I2 = 10, 30
        J1, J2 = 10, 30
        d['LGR3'] = torch_regrid(d['LGR2'][I1:I2, J1:J2, :], [40, 40, 50])

        I1, I2 = 16, 24
        J1, J2 = 16, 24
        d['LGR4'] = torch_regrid(d['LGR3'][I1:I2, J1:J2, :], [40, 40, 50])
        
        d['LGR1'] = d['LGR1'][None,...]
        d['LGR2'] = d['LGR2'][None,...]
        d['LGR3'] = d['LGR3'][None,...]
        d['LGR4'] = d['LGR4'][None,...]
        DICT[f'WELL{int(i_well+1)}'] = d
    DICT['GLOBAL'] = DICT['GLOBAL'][None,...]
    return DICT

def inj_location_dict(pcolor_grid_dict, WELL_LIST):
    INJ_LOCATION_DICT = {}
    for well in WELL_LIST:
        well_x = pcolor_grid_dict[well]['LGR4']['grid_x'][20,20,0]
        well_y = pcolor_grid_dict[well]['LGR4']['grid_y'][20,20,0]
        INJ_LOCATION_DICT[well]=well_x, well_y
    return INJ_LOCATION_DICT

def grid_dict(wells, grid_dict):
    lgrs = list(grid_dict[wells[0]].keys())
    parents = parents = ['GLOBAL'] + lgrs[:-1]
    
    # GLOBAL grid
    grid_x = np.linspace(grid_dict['GLOBAL']['DX']/2,
                         160000 - grid_dict['GLOBAL']['DX']/2,
                         grid_dict['GLOBAL']['NX'])
    grid_y = np.linspace(grid_dict['GLOBAL']['DY']/2,
                         160000-grid_dict['GLOBAL']['DY']/2,
                         grid_dict['GLOBAL']['NY'])
    grid_z = np.linspace(grid_dict['GLOBAL']['DZ']/2,
                         100-grid_dict['GLOBAL']['DZ']/2,
                         grid_dict['GLOBAL']['NZ'])

    grid_x, grid_y, grid_z = np.meshgrid(grid_x, grid_y, grid_z,indexing='ij')

    GRID = {}
    GRID['GLOBAL'] = {'grid_x': grid_x, 'grid_y': grid_y, 'grid_z': grid_z }

    ######################### grid for LGR1 #########################
    for well in wells:
        lgr = 'LGR1'
        parent = 'GLOBAL'
        x_start = GRID[parent]['grid_x'][grid_dict[well][lgr]['I1']-1,0,0] - grid_dict[parent]['DX']/2
        x_end = GRID[parent]['grid_x'][grid_dict[well][lgr]['I2']-1,0,0] + grid_dict[parent]['DX']/2

        grid_x = np.linspace(x_start+grid_dict[well][lgr]['DX']/2, 
                             x_end-grid_dict[well][lgr]['DX']/2,
                             grid_dict[well][lgr]['NX'])

        y_start = GRID[parent]['grid_y'][0,grid_dict[well][lgr]['J1']-1,0] - grid_dict[parent]['DY']/2
        y_end = GRID[parent]['grid_y'][0,grid_dict[well][lgr]['J2']-1,0] + grid_dict[parent]['DY']/2

        grid_y = np.linspace(y_start+grid_dict[well][lgr]['DY']/2, 
                             y_end-grid_dict[well][lgr]['DY']/2,
                             grid_dict[well][lgr]['NY'])

        z_start = GRID[parent]['grid_z'][0,0,grid_dict[well][lgr]['K1']-1] - grid_dict[parent]['DZ']/2
        z_end = GRID[parent]['grid_z'][0,0,grid_dict[well][lgr]['K2']-1] + grid_dict[parent]['DZ']/2

        grid_z = np.linspace(z_start+grid_dict[well][lgr]['DZ']/2, 
                             z_end-grid_dict[well][lgr]['DZ']/2,
                             grid_dict[well][lgr]['NZ'])

        grid_x, grid_y, grid_z = np.meshgrid(grid_x, grid_y, grid_z,indexing='ij')
        GRID[well] = {lgr: {'grid_x': grid_x, 'grid_y': grid_y, 'grid_z': grid_z }}

    ######################### grid for LGR2 and up #########################

    lgrs = lgrs[1:]
    parents = parents[1:]

    for well in wells:
        for i in range(len(lgrs)):
            lgr = lgrs[i]
            parent = parents[i]

            x_start = GRID[well][parent]['grid_x'][grid_dict[well][lgr]['I1']-1,0,0] - grid_dict[well][parent]['DX']/2
            x_end = GRID[well][parent]['grid_x'][grid_dict[well][lgr]['I2']-1,0,0] + grid_dict[well][parent]['DX']/2

            grid_x = np.linspace(x_start+grid_dict[well][lgr]['DX']/2, 
                                 x_end-grid_dict[well][lgr]['DX']/2,
                                 grid_dict[well][lgr]['NX'])

            y_start = GRID[well][parent]['grid_y'][0,grid_dict[well][lgr]['J1']-1,0] - grid_dict[well][parent]['DY']/2
            y_end = GRID[well][parent]['grid_y'][0,grid_dict[well][lgr]['J2']-1,0] + grid_dict[well][parent]['DY']/2

            grid_y = np.linspace(y_start+grid_dict[well][lgr]['DY']/2, 
                                 y_end-grid_dict[well][lgr]['DY']/2,
                                 grid_dict[well][lgr]['NY'])

            z_start = GRID[well][parent]['grid_z'][0,0,grid_dict[well][lgr]['K1']-1] - grid_dict[well][parent]['DZ']/2
            z_end = GRID[well][parent]['grid_z'][0,0,grid_dict[well][lgr]['K2']-1] + grid_dict[well][parent]['DZ']/2

            grid_z = np.linspace(z_start+grid_dict[well][lgr]['DZ']/2, 
                                 z_end-grid_dict[well][lgr]['DZ']/2,
                                 grid_dict[well][lgr]['NZ'])
            grid_x, grid_y, grid_z = np.meshgrid(grid_x, grid_y, grid_z,indexing='ij')
            GRID[well][lgr] = {'grid_x': grid_x, 'grid_y': grid_y, 'grid_z': grid_z }
    
    return GRID

def pcolor_grid_dict(wells, grid_dict):
    lgrs = list(grid_dict[wells[0]].keys())
    parents = parents = ['GLOBAL'] + lgrs[:-1]
    
    # GLOBAL grid
    grid_x = np.linspace(0, 160000, grid_dict['GLOBAL']['NX']+1)
    grid_y = np.linspace(0, 160000, grid_dict['GLOBAL']['NY']+1)
    grid_z = np.linspace(0, 100, grid_dict['GLOBAL']['NZ']+1)

    grid_x, grid_y, grid_z = np.meshgrid(grid_x, grid_y, grid_z,indexing='ij')
    
    GRID = {}
    GRID['GLOBAL'] = {'grid_x': grid_x, 'grid_y': grid_y, 'grid_z': grid_z }

    ######################### grid for LGR1 #########################
    for well in wells:
        lgr = 'LGR1'
        parent = 'GLOBAL'
        x_start = GRID[parent]['grid_x'][grid_dict[well][lgr]['I1']-1,0,0] 
        x_end = GRID[parent]['grid_x'][grid_dict[well][lgr]['I2'],0,0]

        grid_x = np.linspace(x_start, x_end, grid_dict[well][lgr]['NX']+1)

        y_start = GRID[parent]['grid_y'][0,grid_dict[well][lgr]['J1']-1,0]
        y_end = GRID[parent]['grid_y'][0,grid_dict[well][lgr]['J2'],0]

        grid_y = np.linspace(y_start, y_end, grid_dict[well][lgr]['NY']+1)

        z_start = GRID[parent]['grid_z'][0,0,grid_dict[well][lgr]['K1']-1] 
        z_end = GRID[parent]['grid_z'][0,0,grid_dict[well][lgr]['K2']]

        grid_z = np.linspace(z_start, z_end, grid_dict[well][lgr]['NZ']+1)

        grid_x, grid_y, grid_z = np.meshgrid(grid_x, grid_y, grid_z,indexing='ij')
        GRID[well] = {lgr: {'grid_x': grid_x, 'grid_y': grid_y, 'grid_z': grid_z }}

    ######################### grid for LGR2 and up #########################

    lgrs = lgrs[1:]
    parents = parents[1:]

    for well in wells:
        for i in range(len(lgrs)):
            lgr = lgrs[i]
            parent = parents[i]

            x_start = GRID[well][parent]['grid_x'][grid_dict[well][lgr]['I1']-1,0,0] 
            x_end = GRID[well][parent]['grid_x'][grid_dict[well][lgr]['I2'],0,0] 

            grid_x = np.linspace(x_start, x_end, grid_dict[well][lgr]['NX']+1)

            y_start = GRID[well][parent]['grid_y'][0,grid_dict[well][lgr]['J1']-1,0] 
            y_end = GRID[well][parent]['grid_y'][0,grid_dict[well][lgr]['J2'],0]

            grid_y = np.linspace(y_start, y_end, grid_dict[well][lgr]['NY']+1)

            z_start = GRID[well][parent]['grid_z'][0,0,grid_dict[well][lgr]['K1']-1] 
            z_end = GRID[well][parent]['grid_z'][0,0,grid_dict[well][lgr]['K2']] 

            grid_z = np.linspace(z_start, z_end, grid_dict[well][lgr]['NZ']+1)
            grid_x, grid_y, grid_z = np.meshgrid(grid_x, grid_y, grid_z,indexing='ij')
            GRID[well][lgr] = {'grid_x': grid_x, 'grid_y': grid_y, 'grid_z': grid_z }
    
    return GRID

def input_dict(p, temp, inj_rate_dict):
    INPUT_PARAM = {'temp': temp,
                    'p': p,
                    'inj': inj_rate_dict}
    return INPUT_PARAM