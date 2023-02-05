def LiqFrac_Glob(image, Nz,Nr, crop=None, Mask=None):
    """
    Return the global liquid fraction of an image
    
    :param image: 3D image 
    :type image: int numpy array
    :param crop: Study crop region inside the image
    :type crop: [zmin, zmax, ymin, ymax, xmin, xmax] int array
    :param Mask: 3D image 
    :type Mask: int numpy array
    :return: int numpy array
    """    
    
    import numpy as np
    
    # if crop image
    if len(np.shape(crop))>0:
        image=image[crop[0]:crop[1],crop[2]:crop[3],crop[4]:crop[5]]
        if len(np.shape(Mask))>0:
            Mask=Mask[crop[0]:crop[1],crop[2]:crop[3],crop[4]:crop[5]]
    # if mask image
    if len(np.shape(Mask))>0:
        image = image+2*(1-Mask)
        
    val, count = np.unique(image, return_counts=True)
    return count[0]/(count[0]+count[1])


def LiqFrac_CartesMesh(image, Nz,Ny,Nx, crop=None, Mask=None, structured=True):
    """
    Return a 3D zyx grid with its corresponding non-overlapping subvolume (cuboids) liquid fraction 
    
    :param image: 3D image 
    :type image: int numpy array
    :param Nz: number of sub-regions along z
    :type Nz: int
    :param Ny: number of sub-regions along y
    :type Ny: int
    :param Nx: number of sub-regions along x
    :type Nx: int
    :param crop: Study crop region inside the image
    :type crop: [zmin, zmax, ymin, ymax, xmin, xmax] int array
    :param Mask: 3D image 
    :type Mask: int numpy array
    :return: int numpy array
    """    
    
    import numpy as np
    
    # if crop image
    if len(np.shape(crop))>0:
        image=image[crop[0]:crop[1],crop[2]:crop[3],crop[4]:crop[5]]
        if len(np.shape(Mask))>0:
            Mask=Mask[crop[0]:crop[1],crop[2]:crop[3],crop[4]:crop[5]]
    # if mask the image
    if len(np.shape(Mask))>0:
        image = image+2*(1-Mask)
        
    Z,Y,X = np.shape(image)
    zr = np.linspace(0,Z,Nz+1, dtype='uint16')
    yr = np.linspace(0,Y,Ny+1, dtype='uint16')
    xr = np.linspace(0,X,Nx+1, dtype='uint16')
    
    
    if structured:
        Mliqfrac = np.zeros((Nz,Ny,Nx))
        Mgridz = np.zeros((Nz,Ny,Nx))
        Mgridy = np.zeros((Nz,Ny,Nx))
        Mgridx = np.zeros((Nz,Ny,Nx))
    else:
        Mliqfrac = []
        Mgridz = []
        Mgridy = []
        Mgridx = []
        
    for zi in range(len(zr)-1):
        zbeg = zr[zi]
        zend = zr[zi+1]
        for yi in range(len(yr)-1):
            ybeg = yr[yi]
            yend = yr[yi+1]
            for xi in range(len(xr)-1):
                xbeg = xr[xi]
                xend = xr[xi+1]
                val, count = np.unique(image[zbeg:zend,ybeg:yend,xbeg:xend], return_counts=True)
                
                #Liquid fraction
                count0=0; count1=0; count2=0
                for vi in range(len(val)):
                    if val[vi] == 0:
                        count0=count[vi]
                    if val[vi] == 1:
                        count1=count[vi]
                    if val[vi] == 2:
                        count2=count[vi]
                    
                if structured:
                    if count0>0 and count1>0:
                        Mliqfrac[zi,yi,xi] = count0/(count0+count1)
                    elif count0==0 and count1>0:
                        Mliqfrac[zi,yi,xi] = 0
                    elif count0>0 and count1==0:
                        Mliqfrac[zi,yi,xi] = 1    
                    else:
                        Mliqfrac[zi,yi,xi] = 2
                    #Grid
                    Mgridz[zi,yi,xi] = (zbeg+zend)//2
                    Mgridy[zi,yi,xi] = (ybeg+yend)//2
                    Mgridx[zi,yi,xi] = (xbeg+xend)//2
                else:
                    if count0>0 and count1>0:
                        Mliqfrac.append(count0/(count0+count1))
                    elif count0==0 and count1>0:
                        Mliqfrac.append(0)
                    elif count0>0 and count1==0:
                        Mliqfrac.append(1)
                    else:
                        Mliqfrac.append(2)
                    #Grid
                    Mgridz.append((zbeg+zend)//2)
                    Mgridy.append((ybeg+yend)//2)
                    Mgridx.append((xbeg+xend)//2)
                    
    return [Mgridz,Mgridy,Mgridx], Mliqfrac



def LiqFrac_Batch(nameread, namesave, dirread, dirsave, imrange, TypeGrid='Global', Nz=None,Ny=None,Nx=None,Nr=None,Naz=None, crop=None, Mask=False, verbose=False, endread='.tiff', endsave='.pkl', n0=3, structured=True):
    """
    Read 3D binary images and save liquid fraction informations in series (for loop). Save liquid fraction dictionary as pickle: {"crop", "1D, 2D or 3D grid","lf"}
    
    :param series: series name
    :type series: str
    :param series: read image directory
    :type series: str
    :param savedir: save liquid fraction directory
    :type savedir: str
    :param imrange: image index range
    :type imrange: int numpy array
    :param TypeGrid: Optional, type of method: 'Global' for global liquid fraction, 'CartesMesh' for cartesian grid liquid fraction, 'CylMesh' for cylindrical grid liquid fraction
    :type TypeGrid: str
    :param Nz: Optional, number of sub-regions along z
    :type Nz: int
    :param Ny: Optional, number of sub-regions along y
    :type Ny: int
    :param Nx: Optional, number of sub-regions along x
    :type Nx: int
    :param Nr: Optional, number of sub-regions along r
    :type Nr: int
    :param crop: Optional, Study crop region inside the image
    :type crop: [zmin, zmax, ymin, ymax, xmin, xmax] int array
    :param Mask: Optional, 3D image 
    :type Mask: int numpy array
    :param verbose: if True or 1, print progress image by image. If 10, print additionally the extracted liquid fraction dictionary 
    :type verbose: int numpy array
    :return: None
    """
    
    import numpy as np
    from tifffile import imread, imsave
    import pickle as pkl
    import os
    
    from FoamQuant.FromBinary import LiqFrac_Glob
    from FoamQuant.FromBinary import LiqFrac_CartesMesh
    #from FoamQuant.FromBinary.LiqFrac_CylMesh import LiqFrac_CylMesh
    #from FoamQuant.FromBinary.LiqFrac_SpheMesh import LiqFrac_SpheMesh
    from FoamQuant.Helper import strindex
    
    #Check saving directory
    isExist = os.path.exists(dirsave)
    print('Path exist:', isExist)
    if not isExist:
        print('Error: saving path does not exist', dirsave)
        return
       
    for imi in imrange:
        # image string index
        imifordir = strindex(imi, n0)
        # read image
        image = np.asarray(imread(dirread + nameread + imifordir + endread), dtype='uint8')
        
        # if first image? and mask given?
        if imi==imrange[0]:
            if len(np.shape(Mask))==0:
                if not Mask:
                    Mask = np.ones(np.shape(image))
                else:
                    from FoamQuant.Process.MaskCyl import MaskCyl
                    Mask = MaskCyl(image)
        
        # Type of grid for the liquid fraction
        if TypeGrid == 'Global':
            lf = LiqFrac_Glob(image, Mask, crop)
            Pack = {"crop": crop, "lf": lf}
            if verbose == 10:
                print('Liquid fraction image '+str(imi)+': done\ncrop:'+str(crop)+
                      '\nLiqFrac:'+str(lf))
            
        elif TypeGrid == 'CartesMesh':
            grid, lf = LiqFrac_CartesMesh(image, Nz,Ny,Nx, crop, Mask, structured=structured)
            Pack = {"crop": crop, "zgrid": grid[0], "ygrid": grid[1],"xgrid": grid[2],"lf": lf}
            if verbose == 10:
                print('Liquid fraction image '+str(imi)+': done\ncrop:'+str(crop)+
                      '\nzgrid:'+str(grid[0])+
                      '\nygrid:'+str(grid[1])+
                      '\nxgrid:'+str(grid[2])+
                      '\nLiqFrac:'+str(lf)+'\n')
            
        elif TypeGrid == 'CylMesh':
            grid, lf = LiqFrac_CylMesh(image, Nz,Nr,Naz, crop, Mask)
            Pack = {"crop": crop, "zgrid": grid[0], "azigrid": grid[1],"rgrid": grid[2],"lf": lf}
            if verbose == 10:
                print('Liquid fraction image '+str(imi)+': done\ncrop:'+str(crop)+
                      '\nzgrid:'+str(grid[0])+
                      '\nazigrid:'+str(grid[1])+
                      '\nrgrid:'+str(grid[2])+
                      '\nLiqFrac:'+str(lf)+'\n')
                
        elif TypeGrid == 'SpheMesh':
            grid, lf = LiqFrac_SpheMesh(image, Npol,Naz,Nr, crop, Mask)
            Pack = {"crop": crop, "polgrid": grid[0], "azigrid": grid[0], "rgrid": grid[1],"lf": lf}
            if verbose == 10:
                print('Liquid fraction image '+str(imi)+': done\ncrop:'+str(crop)+
                      '\polgrid:'+str(grid[0])+
                      '\nazigrid:'+str(grid[1])+
                      '\nrgrid:'+str(grid[2])+
                      '\nLiqFrac:'+str(lf)+'\n')
        
        # Save as pickle
        with open(dirsave + namesave + imifordir + endsave,'wb') as file:
            pkl.dump(Pack, file, pkl.HIGHEST_PROTOCOL)
        
        # if verbose
        if verbose >= 1:
            print(namesave+imifordir+': done')
        
