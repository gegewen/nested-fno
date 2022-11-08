import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

def plot_x_slice(x,WELL_LIST, LGR_LIST, PCOLOR_GRID_DICT, GRID_IDX_DICT,OUTPUT_DICT,cmin,cmax,
                 xmin=None,xmax=None,ymin=None,ymax=None, figsize=None, title=None, boundary_on=True, grid_width=0):
    xidx = int(x/1600)
    plt.pcolormesh(PCOLOR_GRID_DICT['GLOBAL']['grid_y'][xidx,:,:],
               PCOLOR_GRID_DICT['GLOBAL']['grid_z'][xidx,:,:],
               OUTPUT_DICT['GLOBAL'][-1,xidx,:,:],
               shading='flat', edgecolor='k',linewidth=grid_width)
    plt.clim([cmin, cmax])
    for well in WELL_LIST:
        for lgr in LGR_LIST:
            lgr_start = np.min(PCOLOR_GRID_DICT[well][lgr]['grid_x'])
            lgr_end = np.max(PCOLOR_GRID_DICT[well][lgr]['grid_x'])
            if (x>lgr_start) and (x<lgr_end):
                xidx = int((x - lgr_start)/GRID_IDX_DICT[well][lgr]['DY'])
                plt.pcolormesh(PCOLOR_GRID_DICT[well][lgr]['grid_y'][xidx,:,:],
                           PCOLOR_GRID_DICT[well][lgr]['grid_z'][xidx,:,:],
                           OUTPUT_DICT[well][lgr][-1,xidx,:,:], shading='flat', edgecolor='k',linewidth=grid_width)
                plt.clim([cmin, cmax])
                lgr_start = np.min(PCOLOR_GRID_DICT[well][lgr]['grid_y'][0,:,:])
                lgr_end = np.max(PCOLOR_GRID_DICT[well][lgr]['grid_y'][0,:,:])
                if boundary_on:
                    ax = plt.gca()
                    rect = Rectangle((lgr_start,100),
                                      (lgr_end-lgr_start),-100,linewidth=1,edgecolor='r',linestyle = '-',facecolor='none')
                    ax.add_patch(rect)
                
            elif (x==lgr_start):
                xidx = 0
                plt.pcolormesh(PCOLOR_GRID_DICT[well][lgr]['grid_y'][xidx,:,:],
                           PCOLOR_GRID_DICT[well][lgr]['grid_z'][xidx,:,:],
                           OUTPUT_DICT[well][lgr][-1,xidx,:,:], shading='flat', edgecolor='k',linewidth=grid_width)
                plt.clim([cmin, cmax])
                lgr_start = np.min(PCOLOR_GRID_DICT[well][lgr]['grid_y'][0,:,:])
                lgr_end = np.max(PCOLOR_GRID_DICT[well][lgr]['grid_y'][0,:,:])
                if boundary_on:
                    ax = plt.gca()
                    rect = Rectangle((lgr_start,100),
                                      (lgr_end-lgr_start),-100,linewidth=1,edgecolor='r',linestyle = '-',facecolor='none')
                    ax.add_patch(rect)
            
    plt.title(f'x = {x} m {title}')
    if xmin is not None:
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
    plt.colorbar(fraction=0.01)
    plt.gca().invert_yaxis()
    
def plot_z_slice(z,WELL_LIST, LGR_LIST, PCOLOR_GRID_DICT, GRID_IDX_DICT,OUTPUT_DICT,cmin,cmax,
                 xmin=None,xmax=None,ymin=None,ymax=None,grid_on=False, title=None, boundary_on=True):
    zidx = int(z/20)
    if grid_on is True:
        plt.pcolormesh(PCOLOR_GRID_DICT['GLOBAL']['grid_x'][:,:,zidx],
                   PCOLOR_GRID_DICT['GLOBAL']['grid_y'][:,:,zidx],
                   OUTPUT_DICT['GLOBAL'][-1,:,:,zidx],
                   shading='flat', edgecolor='k')
    else:
        plt.pcolormesh(PCOLOR_GRID_DICT['GLOBAL']['grid_x'][:,:,zidx],
                   PCOLOR_GRID_DICT['GLOBAL']['grid_y'][:,:,zidx],
                   OUTPUT_DICT['GLOBAL'][-1,:,:,zidx],
                   shading='flat')
    plt.clim([cmin, cmax])

    for well in WELL_LIST:
        for lgr in LGR_LIST:
            zidx = int(z/GRID_IDX_DICT[well][lgr]['DZ'])
            
            if grid_on is True:
                plt.pcolormesh(PCOLOR_GRID_DICT[well][lgr]['grid_x'][:,:,zidx],
                           PCOLOR_GRID_DICT[well][lgr]['grid_y'][:,:,zidx],
                           OUTPUT_DICT[well][lgr][-1,:,:,zidx], shading='flat', edgecolor='k')
            else:
                plt.pcolormesh(PCOLOR_GRID_DICT[well][lgr]['grid_x'][:,:,zidx],
                   PCOLOR_GRID_DICT[well][lgr]['grid_y'][:,:,zidx],
                   OUTPUT_DICT[well][lgr][-1,:,:,zidx], shading='flat')
            plt.clim([cmin, cmax])
            
            if boundary_on:
                dx = np.max(PCOLOR_GRID_DICT[well][lgr]['grid_x'][:,:,zidx]) - np.min(PCOLOR_GRID_DICT[well][lgr]['grid_x'][:,:,zidx])
                dy = np.max(PCOLOR_GRID_DICT[well][lgr]['grid_y'][:,:,zidx])- np.min(PCOLOR_GRID_DICT[well][lgr]['grid_y'][:,:,zidx])
                ax = plt.gca()
                rect = Rectangle((np.min(PCOLOR_GRID_DICT[well][lgr]['grid_x'][:,:,zidx]),
                                np.min(PCOLOR_GRID_DICT[well][lgr]['grid_y'][:,:,zidx])),
                                  dx,dy,linewidth=1,edgecolor='r',linestyle = '-',facecolor='none')
                ax.add_patch(rect)
                             
    if xmin is not None:
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])

    
    plt.title(f'z = {z} m {title}')

    plt.colorbar(fraction=0.01)

def pcolor_grid_dict(wells, grid_dict):
    lgrs = list(grid_dict[wells[0]].keys())
    parents = parents = ['GLOBAL'] + lgrs[:-1]
    
    # GLOBAL grid
    grid_x = np.linspace(0, 100000, grid_dict['GLOBAL']['NX']+1)
    grid_y = np.linspace(0, 100000, grid_dict['GLOBAL']['NY']+1)
    grid_z = np.linspace(0, 200, grid_dict['GLOBAL']['NZ']+1)

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


    
def plot_xz_line(x, z, WELL_LIST, LGR_LIST, GRID_IDX_DICT,OUTPUT_DICT,xmin,xmax):
    xidx = int(x/1000)
    PCOLOR_GRID_DICT = pcolor_grid_dict(WELL_LIST, GRID_IDX_DICT)
    plt.figure(figsize=(15,5))
    active = OUTPUT_DICT['GLOBAL'][-1,xidx,:,z]!=0
    plt.plot(GRID_DICT['GLOBAL']['grid_y'][xidx,:,z][active],
             OUTPUT_DICT['GLOBAL'][-1,xidx,:,z][active],'.')

    for well in WELL_LIST:
        for lgr in LGR_LIST:
            lgr_start = np.min(PCOLOR_GRID_DICT[well][lgr]['grid_x'])
            lgr_end = np.max(PCOLOR_GRID_DICT[well][lgr]['grid_x'])
            if (x>=lgr_start) and (x<lgr_end):
                xidx = (np.abs(PCOLOR_GRID_DICT[well][lgr]['grid_x'][:,0,0] - x)).argmin()
                active = OUTPUT_DICT[well][lgr][-1,xidx,:,z]>1
                plt.plot(GRID_DICT[well][lgr]['grid_y'][xidx,:,z][active],
                         OUTPUT_DICT[well][lgr][-1,xidx,:,z][active],
                         '.',label=well+lgr)
    plt.title(f'x = {x} m')
    plt.xlim([xmin, xmax])
    plt.ylim([190,220])
    plt.legend()
    plt.show()
    
def plot_zglobal_slice(z,WELL_LIST, LGR_LIST, GRID_IDX_DICT,OUTPUT_DICT,cmin,cmax,
                 xmin=None,xmax=None,ymin=None,ymax=None,grid_on=False):
    plt.figure(figsize=(10,9))
    
    PCOLOR_GRID_DICT = pcolor_grid_dict(WELL_LIST, GRID_IDX_DICT)
    if grid_on is True:
        plt.pcolormesh(PCOLOR_GRID_DICT['GLOBAL']['grid_x'][:,:,z],
                   PCOLOR_GRID_DICT['GLOBAL']['grid_y'][:,:,z],
                   OUTPUT_DICT['GLOBAL'][-1,:,:,z],
                   shading='flat', edgecolor='k')
    else:
        plt.pcolormesh(PCOLOR_GRID_DICT['GLOBAL']['grid_x'][:,:,z],
                   PCOLOR_GRID_DICT['GLOBAL']['grid_y'][:,:,z],
                   OUTPUT_DICT['GLOBAL'][-1,:,:,z],
                   shading='flat')
    
    plt.clim([cmin, cmax])
    if xmin is not None:
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])

    plt.colorbar(fraction=0.01)

def plot_y_slice(y,WELL_LIST, LGR_LIST, GRID_IDX_DICT,OUTPUT_DICT,cmin,cmax,
                 xmin=None,xmax=None,ymin=None,ymax=None, title=None):
    yidx = int(y/1000)
    print(yidx)
    PCOLOR_GRID_DICT = pcolor_grid_dict(WELL_LIST, GRID_IDX_DICT)
    plt.pcolormesh(PCOLOR_GRID_DICT['GLOBAL']['grid_x'][:,yidx,:],
               PCOLOR_GRID_DICT['GLOBAL']['grid_z'][:,yidx,:],
               OUTPUT_DICT['GLOBAL'][-1,:,yidx,:],
               shading='flat')
    plt.clim([cmin, cmax])

    for well in WELL_LIST:
        for lgr in LGR_LIST:
            lgr_start = np.min(PCOLOR_GRID_DICT[well][lgr]['grid_y'])
            lgr_end = np.max(PCOLOR_GRID_DICT[well][lgr]['grid_y'])
            if (y>lgr_start) and (y<lgr_end):
                yidx = int((y - lgr_start)/GRID_IDX_DICT[well][lgr]['DY'])
                plt.pcolormesh(PCOLOR_GRID_DICT[well][lgr]['grid_x'][:,yidx,:],
                           PCOLOR_GRID_DICT[well][lgr]['grid_z'][:,yidx,:],
                           OUTPUT_DICT[well][lgr][-1,:,yidx,:], shading='flat')
                plt.clim([cmin, cmax])
            elif (y==lgr_start):
                yidx = 0
                plt.pcolormesh(PCOLOR_GRID_DICT[well][lgr]['grid_x'][:,yidx,:],
                           PCOLOR_GRID_DICT[well][lgr]['grid_z'][:,yidx,:],
                           OUTPUT_DICT[well][lgr][-1,:,yidx,:], shading='flat')
                plt.clim([cmin, cmax])
            
    plt.title(f'y = {y} m {title}')
    if xmin is not None:
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
    plt.colorbar(fraction=0.01)
    plt.gca().invert_yaxis()
    plt.show()
    
    
    
    
    
    
