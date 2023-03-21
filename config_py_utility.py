import numpy as np
import torch.nn.functional as F
import torch

def return_upsample_dict(OUTPUT_DICT, WELL_LIST, GRID_IDX_DICT):
    OUTPUT_UPSAMPLE_DICT = {} 

    LGR_BEFORE = ['LGR3', 'LGR2', 'LGR1']
    LGR_AFTER = ['LGR4', 'LGR3', 'LGR2']

    for well in WELL_LIST:
        OUTPUT_UPSAMPLE_DICT[well] = {'LGR4': OUTPUT_DICT[well]['LGR4']}
        for iii in range(3):
            lgr_before = LGR_BEFORE[iii]
            lgr_after = LGR_AFTER[iii]

            upsampled = np.copy(OUTPUT_DICT[well][lgr_before][-1,:,:,:])
            nx_new = GRID_IDX_DICT[well][lgr_after]['I2'] - GRID_IDX_DICT[well][lgr_after]['I1'] + 1
            ny_new = GRID_IDX_DICT[well][lgr_after]['J2'] - GRID_IDX_DICT[well][lgr_after]['J1'] + 1
            nz_new = GRID_IDX_DICT[well][lgr_after]['K2'] - GRID_IDX_DICT[well][lgr_after]['K1'] + 1

            A = F.interpolate(torch.from_numpy(OUTPUT_UPSAMPLE_DICT[well][lgr_after][-1,:,:,:])[None, None,...], 
                              size=[nx_new,ny_new,nz_new], mode='trilinear', align_corners=False)[0,0,...].numpy()

            upsampled[GRID_IDX_DICT[well][lgr_after]['I1']-1:GRID_IDX_DICT[well][lgr_after]['I2'],
                      GRID_IDX_DICT[well][lgr_after]['J1']-1:GRID_IDX_DICT[well][lgr_after]['J2'],:] = A

            if well in OUTPUT_UPSAMPLE_DICT:
                    OUTPUT_UPSAMPLE_DICT[well].update({lgr_before: upsampled[None,...]})
            else:
                OUTPUT_UPSAMPLE_DICT[well]={lgr_before: upsampled[None,...]}

    upsampled = np.copy(OUTPUT_DICT['GLOBAL'][-1,:,:,:])
    for well in WELL_LIST:
        nx_new = GRID_IDX_DICT[well]['LGR1']['I2'] - GRID_IDX_DICT[well]['LGR1']['I1'] + 1
        ny_new = GRID_IDX_DICT[well]['LGR1']['J2'] - GRID_IDX_DICT[well]['LGR1']['J1'] + 1
        nz_new = GRID_IDX_DICT[well]['LGR1']['K2'] - GRID_IDX_DICT[well]['LGR1']['K1'] + 1
        A = F.interpolate(torch.from_numpy(OUTPUT_UPSAMPLE_DICT[well]['LGR1'][-1,:,:,:])[None, None,...], 
                              size=[nx_new,ny_new,nz_new],  mode='trilinear', align_corners=False)[0,0,...].numpy()
        upsampled[GRID_IDX_DICT[well]['LGR1']['I1']-1:GRID_IDX_DICT[well]['LGR1']['I2'],
                   GRID_IDX_DICT[well]['LGR1']['J1']-1:GRID_IDX_DICT[well]['LGR1']['J2'],:] = A
    OUTPUT_UPSAMPLE_DICT['GLOBAL'] = upsampled
    
    return OUTPUT_UPSAMPLE_DICT

def return_OUTPUT_DICT(meta_data, case_name):
    nt = list(meta_data[case_name]['data'].keys())
    OUT = {}
    GRID_IDX_DICT = meta_data[case_name]['GRID_IDX_DICT']
    WELL_LIST = meta_data[case_name]['WELL_LIST']
    LGR_LIST = meta_data[case_name]['LGR_LIST']

    for name in [ 'BGSAT', 'BPR']:
        out = {}
        lname = f'L{name}'
        for t in nt:
            data = meta_data[case_name]['data'][t]
            output_dict = {}

            output_dict['GLOBAL'] = data[name].reshape((-1, GRID_IDX_DICT['GLOBAL']['NX'],
                                                         GRID_IDX_DICT['GLOBAL']['NY'],
                                                         GRID_IDX_DICT['GLOBAL']['NZ']))
            N_LIST = [0]
            idx = 0
            for well in WELL_LIST:
                for lgr in LGR_LIST:
                    n_prev = N_LIST[idx]
                    idx += 1
                    n_cur = n_prev+GRID_IDX_DICT[well][lgr]['NX'] * GRID_IDX_DICT[well][lgr]['NY'] * GRID_IDX_DICT[well][lgr]['NZ']
                    N_LIST.append(n_cur)

                    if well in output_dict:
                        output_dict[well].update({lgr: data[lname][:,n_prev: n_cur].reshape(-1,
                                                                                             GRID_IDX_DICT[well][lgr]['NX'],
                                                                                             GRID_IDX_DICT[well][lgr]['NY'],
                                                                                             GRID_IDX_DICT[well][lgr]['NZ']) })
                    else:
                        output_dict[well] = {lgr: data[lname][:,n_prev: n_cur].reshape(-1,
                                                                                     GRID_IDX_DICT[well][lgr]['NX'],
                                                                                     GRID_IDX_DICT[well][lgr]['NY'],
                                                                                     GRID_IDX_DICT[well][lgr]['NZ']) }
            out[t] = output_dict
        OUT[name] = out

    out = {}
    for t in nt:
        output_dict = {}
        output_dict['GLOBAL'] = OUT['BPR'][t]['GLOBAL'] - OUT['BPR'][0]['GLOBAL']

        for well in WELL_LIST:
            for lgr in LGR_LIST:
                if well in output_dict:
                    output_dict[well].update({lgr:  OUT['BPR'][t][well][lgr] - OUT['BPR'][0][well][lgr]})
                else:
                    output_dict[well] = {lgr:  OUT['BPR'][t][well][lgr] - OUT['BPR'][0][well][lgr]}
        out[t] = output_dict
    OUT['dP'] = out

    out = {}
    for t in nt:
        output_dict = {}
        output_dict['GLOBAL'] = OUT['dP'][t]['GLOBAL'] > 0.1

        for well in WELL_LIST:
            for lgr in LGR_LIST:
                if well in output_dict:
                    output_dict[well].update({lgr:  OUT['dP'][t][well][lgr] > 0.1})
                else:
                    output_dict[well] = {lgr:  OUT['dP'][t][well][lgr] > 0.1 }
        out[t] = output_dict
    OUT['P_influence'] = out
    return OUT


def return_upsample_all_time(OUTPUT_DICT, name, WELL_LIST, GRID_IDX_DICT,LGR_LIST):
    OUT = {}
    OUT['GLOBAL'] = np.zeros((1, 100, 100, 5, 24))
    for well in WELL_LIST:
        for lgr in LGR_LIST:
            nx, ny, nz = GRID_IDX_DICT[well][lgr]['NX'], GRID_IDX_DICT[well][lgr]['NY'], GRID_IDX_DICT[well][lgr]['NZ']
            if well in OUT:
                OUT[well].update({lgr: np.zeros((1,nx,ny,nz,24))})
            else:
                OUT[well]={lgr: np.zeros((1,nx,ny,nz,24))}
                
    for t in range(1,25):
        up_sample_dict = return_upsample_dict(OUTPUT_DICT, t, name, WELL_LIST, GRID_IDX_DICT)
        OUT['GLOBAL'][0,:,:,:,t-1] = up_sample_dict['GLOBAL']
        
        for well in WELL_LIST:
            for lgr in LGR_LIST:
                OUT[well][lgr][0,:,:,:,t-1] = up_sample_dict[well][lgr]
        
    return OUT

def return_inj_map_dict(well_list,rate_dict,inj_loc_dict,center_dict, LGR_LIST):
    inj_norm = lambda x: (x)/(2942777.68785957)

    INJ_MAP_DICT = {}

    inj_map = np.zeros(center_dict['GLOBAL']['grid_x'].shape)
    for well in well_list:
        well_x, well_y = inj_loc_dict[well]
        xidx = (np.abs(center_dict['GLOBAL']['grid_x'][:,0,0] - well_x)).argmin()
        yidx = (np.abs(center_dict['GLOBAL']['grid_y'][0,:,0] - well_y)).argmin()
        inj_map[xidx, yidx, :] = inj_norm(rate_dict[well])
        INJ_MAP_DICT['GLOBAL'] = inj_map

    for well in well_list:
        well_x, well_y = inj_loc_dict[well]
        for lgr in LGR_LIST:
            inj_map = np.zeros(center_dict[well][lgr]['grid_x'].shape)
            xidx = (np.abs(center_dict[well][lgr]['grid_x'][:,0,0] - well_x)).argmin()
            yidx = (np.abs(center_dict[well][lgr]['grid_y'][0,:,0] - well_y)).argmin()
            inj_map[xidx, yidx, :] = inj_norm(rate_dict[well])
            if well in INJ_MAP_DICT:
                INJ_MAP_DICT[well].update({lgr: inj_map})
            else:
                INJ_MAP_DICT[well]={lgr: inj_map}
            
    return INJ_MAP_DICT