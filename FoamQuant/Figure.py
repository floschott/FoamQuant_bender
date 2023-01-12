def Cut3D(image, zcut=False, ycut=False, xcut=False, showcuts=False, showaxes=False, showhistogram=False,
          histtitle=False, cmap='gray', interpolation=None, figblocksize=5, returnfig=False, returnhistogram=False, vmin=None, vmax=None, printminmax=False):
    """
    Plot a 3x1 figure showing three orthogonal cross-sections of the 3D image.
    
    :param image: 3D numpy array
    :type image: int
    :param zcut: Optional z cut value
    :type zcut: int or False
    :param ycut: Optional y cut value
    :type ycut: int or False
    :param xcut: Optional x cut value
    :type xcut: int or False 
    :param showcuts: Optional plot the orthogonal cuts
    :type showcuts: Bool
    :param showaxes: Optional plot the axes
    :type showaxes: Bool
    :param showhistogram: Optional add a grey-scale histogram, the figure become 3x1
    :type showhistogram: Bool
    :param histtitle: Optional add a tittle to the histogram
    :type histtitle: str or False
    :param cmap: Optional the color map used for the cuts, Default cmap = 'gray' 
    :type cmap: str or cmap type
    :param interpolation: Optional type of interpolation, Default interpolation = None 
    :type interpolation: str or None
    :param figblocksize: Optional size of the subfigure, Default figblocksize = 5 
    :type figblocksize: float
    :param returnfig: Optional, if should return the figure, if not returns None
    :type returnfig: Bool
    :param vmin: Optional, min value for the color range
    :type vmin: Bool
    :param vmax: Optional, max value for the color range
    :type vmax: Bool
    :param printminmax: Optional, print min and max for the whole image and the three projections
    :type printminmax: Bool
    :return: None or fig type
    """    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.exposure import histogram
    
    shapezyx = np.shape(image)
    if zcut == False:
        zcut = shapezyx[0]//2
    if ycut == False:
        ycut = shapezyx[1]//2
    if xcut == False:
        xcut = shapezyx[2]//2
    
    if showhistogram:
        from skimage.exposure import histogram
        fig, ax = plt.subplots(ncols=4, figsize=(4*figblocksize, figblocksize))
        hist, hist_centers = histogram(image)
        ax[3].plot(hist_centers, hist, lw=2)
        ax[3].set_yscale('log')
        if histtitle != False:
            ax[3].set_title(histtitle)
        plt.tight_layout()
    else:
        fig, ax = plt.subplots(ncols=3, figsize=(3*figblocksize, figblocksize))
        
    if vmin!=None and vmax!=None:
        print('vmin =',vmin, 'vmax =',vmax) 
        ax[0].imshow(image[zcut,:,:], cmap=cmap, interpolation=interpolation, vmin=vmin, vmax=vmax)
        ax[1].imshow(image[:,ycut,:], cmap=cmap, interpolation=interpolation, vmin=vmin, vmax=vmax)
        ax[2].imshow(image[:,:,xcut], cmap=cmap, interpolation=interpolation, vmin=vmin, vmax=vmax)
    else:    
        ax[0].imshow(image[zcut,:,:], cmap=cmap, interpolation=interpolation)
        ax[1].imshow(image[:,ycut,:], cmap=cmap, interpolation=interpolation)
        ax[2].imshow(image[:,:,xcut], cmap=cmap, interpolation=interpolation)
    
    if showcuts:
        ax[0].plot([1,shapezyx[2]-1,shapezyx[2]-1,1,1],[shapezyx[1]-1,shapezyx[1]-1,1,1,shapezyx[1]-1],'r',linewidth=3) #zcut
        ax[1].plot([1,shapezyx[2]-1,shapezyx[2]-1,1,1],[shapezyx[0]-1,shapezyx[0]-1,1,1,shapezyx[0]-1],'b',linewidth=3) #ycut
        ax[2].plot([1,shapezyx[1]-1,shapezyx[1]-1,1,1],[shapezyx[0]-1,shapezyx[0]-1,1,1,shapezyx[0]-1],'g',linewidth=3) #xcut
        
        ax[0].plot([1,shapezyx[2]-1],[ycut,ycut],'b',linewidth=3) #ycut
        ax[0].plot([xcut,xcut],[1,shapezyx[1]-1],'g',linewidth=3) #xcut
        
        ax[1].plot([1,shapezyx[2]-1],[zcut,zcut],'r',linewidth=3) #zcut
        ax[1].plot([xcut,xcut],[1,shapezyx[0]-1],'g',linewidth=3) #xcut
        
        ax[2].plot([1,shapezyx[1]-1],[zcut,zcut],'r',linewidth=3) #zcut
        ax[2].plot([ycut,ycut],[1,shapezyx[0]-1],'b',linewidth=3) #ycut
    
    if showaxes:
        ax[0].set_xlabel('x'); ax[0].set_ylabel('y')
        ax[1].set_xlabel('x'); ax[1].set_ylabel('z')
        ax[2].set_xlabel('y'); ax[2].set_ylabel('z')
        
    plt.tight_layout()
    
    if printminmax:
        print('MIN:',np.nanmin(image),'MAX:',np.nanmax(image))
    
    if returnfig:
        return fig
    
    if returnhistogram:
        return hist, hist_centers  
    
def Proj3D(image, showaxes=False, cmap='gray', interpolation=None, figblocksize=5, returnfig=False, vmin=None, vmax=None, printminmax=False):
    """
    Plot a 3x1 figure showing three orthogonal projections of the 3D image.
    
    :param image: 3D image
    :type image: numpy array
    :param showaxes: Optional plot the axes
    :type showaxes: Bool
    :param cmap: Optional the color map used for the projections, Default cmap = 'gray' 
    :type cmap: str or cmap type
    :param interpolation: Optional the type of interpolation, Default interpolation = None 
    :type interpolation: str or None
    :param figblocksize: Optional size of the subfigure, Default figblocksize = 5 
    :type figblocksize: float
    :param returnfig: Optional if should return the figure, if not returns None
    :type returnfig: Bool
    :param vmin: Optional, min value for the color range
    :type vmin: Bool
    :param vmax: Optional, max value for the color range
    :type vmax: Bool
    :param printminmax: Optional, print min and max for the whole image and the three projections
    :type printminmax: Bool
    :return: None or fig
    """
    
    import numpy as np
    import matplotlib.pyplot as plt    
    
   
    fig, ax = plt.subplots(ncols=3, figsize=(3*figblocksize, figblocksize))
        
    if vmin!=None and vmax!=None:
        print('vmin =',vmin, 'vmax =',vmax) 
        ax[0].imshow(np.nanmean(image,0), cmap=cmap, interpolation=interpolation, vmin=vmin, vmax=vmax)
        ax[1].imshow(np.nanmean(image,1), cmap=cmap, interpolation=interpolation, vmin=vmin, vmax=vmax)
        ax[2].imshow(np.nanmean(image,2), cmap=cmap, interpolation=interpolation, vmin=vmin, vmax=vmax)
    else:    
        ax[0].imshow(np.nanmean(image,0), cmap=cmap, interpolation=interpolation)
        ax[1].imshow(np.nanmean(image,1), cmap=cmap, interpolation=interpolation)
        ax[2].imshow(np.nanmean(image,2), cmap=cmap, interpolation=interpolation)
    
    if showaxes:
        ax[0].set_xlabel('x'); ax[0].set_ylabel('y')
        ax[1].set_xlabel('x'); ax[1].set_ylabel('z')
        ax[2].set_xlabel('y'); ax[2].set_ylabel('z')
        
    plt.tight_layout()
    
    if printminmax:
        print('MIN:',np.nanmin(image),'MAX:',np.nanmax(image))
        print('Min0:',np.nanmin(np.nanmean(image,0)),'Min0:',np.nanmax(np.nanmean(image,0)))
        print('Min1:',np.nanmin(np.nanmean(image,1)),'Max1',np.nanmax(np.nanmean(image,1)))
        print('Min2:',np.nanmin(np.nanmean(image,2)),'Max2:',np.nanmax(np.nanmean(image,2)))
    
    if returnfig:
        return fig

def ellipse_plot(Fig, X,Y, Vect, Val, scale_factor = 1, mirror = False):
    """
    Plot internal 2D structured strain field map.
    
    :param Fig: matplotlib.pyplot figure
    :type Fig: figure type
    :param X: structured (N,M) horyzontal positions
    :type X: numpy array
    :param Y: structured (N,M) vertical positions
    :type Y: numpy array
    :param Vect: structured in-plane eigenvectors [U+x,U+y,U-x,U-y], (N,M,4)
    :type Vect: numpy array
    :param Val: structured in-plane eigenvalues [U+,U-,Uphi], (N,M,3) 
    :type Val: numpy array
    :param scale_factor: Optional, by default 1
    :type scale_factor: float
    :param mirror: Optional, if True the plot is horyzontally flipped
    :type mirror: Bool
    :return: None
    """ 
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    for i1 in range(len(X)):
        for i2 in range(len(X[0])):
            x = X[i1][i2]
            y = Y[i1][i2]

            EigVectPlusx, EigVectPlusy, EigVectMinusx, EigVectMinusy = np.asarray(Vect)[:,i1,i2]
            EigValmax, EigValmin, DefAzi = np.asarray(Val)[:,i1,i2]

            isnan = True
            if EigVectPlusx > 0 and EigVectPlusy >= 0:
                beta = np.arctan(np.abs(EigVectPlusy/EigVectPlusx)); isnan = False
            elif EigVectPlusx <= 0 and EigVectPlusy > 0:
                beta = np.arctan(np.abs(EigVectPlusx/EigVectPlusy)) + np.pi/2; isnan = False
            elif EigVectPlusx < 0 and EigVectPlusy <= 0:
                beta = np.arctan(np.abs(EigVectPlusy/EigVectPlusx)) + np.pi; isnan = False
            elif EigVectPlusx >= 0 and EigVectPlusy < 0:
                beta = np.arctan(np.abs(EigVectPlusx/EigVectPlusy)) + 3*np.pi/2; isnan = False
            elif EigVectPlusx == 0:
                beta = np.pi/2; isnan = False
            elif EigVectPlusy == 0:
                beta = np.pi/2; isnan = False

            if isnan == False:
                alphas = np.linspace(0,2,30)*np.pi
                a = scale_factor*abs(EigValmax)
                b = scale_factor*abs(EigValmin)
                if mirror:
                    beta = np.pi - beta
                    x = -x
                xs = x + np.cos(beta)*np.cos(alphas)*a - np.sin(beta)*np.sin(alphas)*b
                ys = y + np.sin(beta)*np.cos(alphas)*a + np.cos(beta)*np.sin(alphas)*b
                LinePlusx = [x + np.cos(beta)*a, x - np.cos(beta)*a]
                LinePlusy = [y + np.sin(beta)*a, y - np.sin(beta)*a]
                LineMinusx = [x + np.cos(beta+np.pi/2)*b, x - np.cos(beta+np.pi/2)*b]
                LineMinusy = [y + np.sin(beta+np.pi/2)*b, y - np.sin(beta+np.pi/2)*b]
                if np.abs(EigValmax) > np.abs(EigValmin):
                    Fig.plot(xs,ys, 'b-', alpha = 1, linewidth = 3)
                else:
                    Fig.plot(xs,ys, 'r-', alpha = 1, linewidth = 3)
                Fig.plot(LinePlusx,LinePlusy, 'k-', alpha = 1, linewidth = 3)
                
def Histogram(image, histtitle=False):
    """
    Plot a 1x1 grey value histogram.
    
    :param image: 3D image.
    :type image: numpy array
    :return: None
    """
    
    from skimage.exposure import histogram
    fig, ax = plt.subplots(ncols=1, figsize=(5, 5))
    hist, hist_centers = skimage.exposure.histogram(image)
    ax.plot(hist_centers, hist, lw=2)
    ax.set_yscale('log')
    if histtitle != False:
        ax.set_title(histtitle)
        


        
# From: https://github.com/delestro/rand_cmap

def RandomCmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True, GiveRange=None):
    """
    Creates a random colormap for matplotlib. Reference: copied from https://github.com/delestro/rand_cmap
    
    :param nlabels: Number of labels (size of colormap)
    :type nlabels: int
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :type type: str
    :param first_color_black: Option to use first color as black, True or False
    :type first_color_black: Bool
    :param last_color_black: Option to use last color as black, True or False
    :type last_color_black: Bool
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :type verbose: Bool
    :return: matplotlib colormap
    """
    
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np


    if type not in ('bright', 'soft','dark','special'):
        print ('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)
        
        
    # Generate dark colors, by limiting the RGB spectrum
    if type == 'dark':
        low = 0.05
        high = 0.4
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)    
        
        
    # Generate ranged colors, by limiting the RGB spectrum
    if type == 'special' and len(GiveRange)>1:
        low = GiveRange[0]
        high = GiveRange[1]
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)    
    

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')
    return random_colormap