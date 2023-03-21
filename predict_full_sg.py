import torch
import numpy as np

def predict_full_sg(input_dict, MODEL_DICT, NORMALIZER_DICT, device):
    with torch.no_grad():
        PRED = {}
        x = input_dict['GLOBAL'].to(device)
        x[...,-1:] = NORMALIZER_DICT['input']['GLOBAL'].encode(x.to(device)[...,-1:])
        pred = NORMALIZER_DICT['output']['GLOBAL'].decode(MODEL_DICT['GLOBAL'](x)).cpu()
        PRED['GLOBAL'] = pred

        WELL_LIST = list(input_dict.keys())
        WELL_LIST.remove('GLOBAL')
        
        for well in WELL_LIST:
            lgr_dict = {}
            x_lgr1 = input_dict[well]['LGR1']
            a = np.abs(input_dict['GLOBAL'][0,0,:,:,0,0][:,0].numpy()-input_dict[well]['LGR1'][0,:,:,0,0,0][:,0].numpy()[0])
            I1 = np.unravel_index(np.argmin(a, axis=None), a.shape)[0] - 15
            a = np.abs(input_dict['GLOBAL'][0,0,:,:,0,0][:,0].numpy()-input_dict[well]['LGR1'][0,:,:,0,0,0][:,0].numpy()[-1])
            I2 = np.unravel_index(np.argmin(a, axis=None), a.shape)[0] + 16
            a = np.abs(input_dict['GLOBAL'][0,0,:,:,0,1][0,:].numpy()-input_dict[well]['LGR1'][0,:,:,0,0,1][0,:].numpy()[0])
            J1 = np.unravel_index(np.argmin(a, axis=None), a.shape)[0] - 15
            a = np.abs(input_dict['GLOBAL'][0,0,:,:,0,1][0,:].numpy()-input_dict[well]['LGR1'][0,:,:,0,0,1][0,:].numpy()[-1])
            J2 = np.unravel_index(np.argmin(a, axis=None), a.shape)[0] + 16
            coarse = np.repeat(PRED['GLOBAL'][0,...][:,I1:I2,J1:J2,:,:],5,axis=-2).permute(-1,1,2,3,0)[...,None]
            x_LGR1 = torch.cat((x_lgr1[...,:-1],coarse),axis=-1)
            x_LGR1 = x_LGR1.permute(0,4,1,2,3,5).to(device)
            x_LGR1[...,-1:] = NORMALIZER_DICT['input']['LGR1'].encode(x_LGR1.to(device)[...,-1:])
            pred = MODEL_DICT['LGR1'](x_LGR1).cpu()
            lgr_dict['LGR1'] = pred
            
            x_lgr2 = input_dict[well]['LGR2']
            coarse = np.repeat(lgr_dict['LGR1'][0,...],2,axis=-2).permute(-1,1,2,3,0)[...,None]
            x_LGR2 = torch.cat((x_lgr2[...,:-1],coarse),axis=-1)
            x_LGR2 = x_LGR2.permute(0,4,1,2,3,5).to(device)
            pred = MODEL_DICT['LGR2'](x_LGR2).cpu()
            lgr_dict['LGR2'] = pred

            x_lgr3 = input_dict[well]['LGR3']
            coarse = lgr_dict['LGR2'][0,...].permute(-1,1,2,3,0)[...,None]
            x_LGR3 = torch.cat((x_lgr3[...,:-1],coarse),axis=-1)
            x_LGR3 = x_LGR3.permute(0,4,1,2,3,5).to(device)
            pred = MODEL_DICT['LGR3'](x_LGR3).cpu()
            lgr_dict['LGR3'] = pred

            x_lgr4 = input_dict[well]['LGR4']
            coarse = lgr_dict['LGR3'][0,...].permute(-1,1,2,3,0)[...,None]
            x_LGR4 = torch.cat((x_lgr4[...,:-1],coarse),axis=-1)
            x_LGR4 = x_LGR4.permute(0,4,1,2,3,5).to(device)
            pred = MODEL_DICT['LGR4'](x_LGR4).cpu()
            lgr_dict['LGR4'] = pred

            PRED[well] = lgr_dict

        PRED['GLOBAL'] *= 0
        return PRED