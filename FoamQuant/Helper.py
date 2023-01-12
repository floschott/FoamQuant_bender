def strindex(i, n0):
    """
    Return str index written on n0 digit
    
    :param i: index 
    :type i: int
    :param n0: number of 0 digit
    :type n0: int
    :return: str index
    """    
    
    istr = str(i)
    istrlen = len(istr)
    fullistr = (n0-istrlen)*'0'+istr
    return fullistr

def RangeList(i1, i2, verbose=False):
    import numpy as np
    List = np.uint8(np.linspace(i1,i2,i2-i1+1))
    if verbose:
        print(List)
    return List

def ReadRaw(series, imi, rawdir, zN=800, top=0, bottom=None):
    import numpy as np
    from tifffile import imread
    
    # image string index
    imistr = str(imi)
    imistrlen = len(imistr)
    imifordir = (5-imistrlen)*'0'+imistr
    if bottom != None:
        zN=bottom
        
    # Init image
    image = np.zeros((zN-top,2016,2016))
    for zi in range(top, zN):
        # horyzontal slice string index
        zistr = str(zi+1)
        zistrlen = len(zistr)
        zifordir = (3-zistrlen)*'0'+zistr
        # horyzontal slice directory
        imdir = rawdir + '/' + series + '/' + 'rec_8bit_phase_'+imifordir + '/' + series+'_'+imifordir+'_'+zifordir+'.rec.8bit.tif'
        # read slice and put into 3D image
        image[zi-top] = imread(imdir)
    # return 3D image
    return image