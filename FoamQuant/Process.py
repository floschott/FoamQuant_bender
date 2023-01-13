def RemoveBackground(image, method='white_tophat', radius=5):
    """
    Remove grey-scale image low frequency background
    
    :param image: 3D image 
    :type image: float numpy array
    :param method: method for removing the background, either 'white_tophat':white tophat filter or 'remove_gaussian': remove the Gaussian filtered image
    :type method: str
    :param radius: white_tophat kernel radius or sigma gaussian filter radius
    :type radius: int
    :return: float numpy array
    """
    
    if method == 'white_tophat':
        from skimage.morphology import white_tophat
        from skimage.morphology import ball
        filtered = white_tophat(image, ball(radius)) # radius
        return filtered
        
    if method == 'remove_gaussian':
        from skimage.filters import gaussian
        import numpy as np
        filtered = gaussian(image, sigma=radius, preserve_range=True) # radius
        filtered = image-filtered
        filtered = (filtered-np.min(filtered))/(np.max(filtered)-np.min(filtered))
        return filtered
    
    
def RemoveSpeckle(image, method='median', radius=1, weight=0.1):
    """
    Remove speckle from the image
    
    :param image: 3D image 
    :type image: float numpy array
    :param method: method for removing the speckle, either 'median', 'gaussian' or 'tv_chambolle'
    :type method: str
    :param radius: kernel radius or sigma gaussian filter radius
    :type radius: int
    :param weight: weight for tv_chambolle
    :type weight: int
    :return: float numpy array
    """
    
    import numpy as np
    if method == 'median':
        from scipy import ndimage
        filtered = ndimage.median_filter(image, size=(radius, radius, radius)) # radius
        return filtered
        
    if method == 'gaussian':
        from skimage.filters import gaussian
        filtered = gaussian(image, sigma=radius) # radius
        return filtered
    
    if method == 'tv_chambolle':
        from skimage.restoration import denoise_tv_chambolle
        filtered = denoise_tv_chambolle(image,weight=weight) # weight
        return filtered


def PhaseSegmentation(image, method='ostu_global', th=0.5, radius=5, th0=0.3, th1=0.7, returnOtsu=False):
    """
    Perform the phase segmentation
    
    :param image: 3D image 
    :type image: float numpy array
    :param method: Optional, method for segmenting the phase, either 'simple' for simple threshold, 'ostu_global' for a global Otsu threshold, 'niblack', 'sauvola', or 'sobel', Default is 'ostu_global'
    :type method: str
    :param th: Optional, given threshold for 'simple' method
    :type th: float
    :param th0: Optional, given low threshold for 'sobel' method
    :type th0: float
    :param th1: Optional, given high threshold for 'sobel' method
    :type th1: float
    :param returnotsu: Optional, if True, returns Otsu threshold for 'ostu_global' method
    :type returnotsu: Bool
    :return: int numpy array and float
    """
    
    import numpy as np
    
    if method == 'simple':
        segmented = image <= th
        return np.asarray(segmented,dtype='uint8')
        
    if method == 'ostu_global':
        from skimage.filters import threshold_otsu
        t_glob = threshold_otsu(image)
        segmented = image <= t_glob
        if returnOtsu:
            return np.asarray(segmented,dtype='uint8'), t_glob
        return np.asarray(segmented,dtype='uint8')
    
    if method == 'niblack':
        from skimage.filters import threshold_niblack
        t_loc = threshold_niblack(image)
        segmented = image <= t_loc
        return np.asarray(segmented,dtype='uint8')
    
    if method == 'sauvola':
        from skimage.filters import threshold_sauvola
        t_loc = threshold_sauvola(image)
        segmented = image <= t_loc
        return np.asarray(segmented,dtype='uint8')
        
    if method == 'sobel':
        from skimage.filters import sobel
        from skimage.segmentation import watershed
        edges = sobel(image)
        markers = np.zeros_like(image) 
        foreground, background = 1, 2
        markers[image < th0] = background
        markers[image > th1] = foreground
        segmented = watershed(edges, markers)
        segmented = segmented > 1
        return np.asarray(segmented,dtype='uint8')
    
    
def MaskCyl(image, rpercent=None):
    """ 
    Create a cylindrical mask of the size of the image
    
    :param image: 3D image 
    :type image: float numpy array
    :return: int numpy array
    """    
    
    import numpy as np
    from spam.mesh.structured import createCylindricalMask
    if rpercent != None:
        cyl = createCylindricalMask(np.shape(image), np.int((np.shape(image)[1]-2)//2*rpercent), voxSize=1.0, centre=None)
    else:
        cyl = createCylindricalMask(np.shape(image), (np.shape(image)[1]-2)//2, voxSize=1.0, centre=None)
    return cyl


def RemoveSpeckleBin(image, RemoveObjects=True, RemoveHoles=True, BinClosing=False, ClosingRadius=None, GiveVolumes=False, Verbose=True, Vminobj=None, Vminhole=None):
    """
    Remove small objects and holes in binary image
    
    :param image: 3D image 
    :type image: int numpy array
    :param RemoveObjects: if True, removes the small objects
    :type RemoveObjects: Bool
    :param RemoveHoles: if True, removes the small holes
    :type RemoveHoles: Bool
    :param BinClosing: if True, perform a binnary closing of radius ClosingRadius
    :type BinClosing: Bool
    :param ClosingRadius: radius of the binnary closing
    :type ClosingRadius: int
    :param GiveVolumes: if True, returns in addition the used min volume thresholds for objects and holes
    :type GiveVolumes: Bool
    :param Verbose: if True, print progression steps of the cleaning
    :type Verbose: Bool
    :param Vminobj: if given the min volume threshold for the objects is not computed, Vminobj is used as the threshold 
    :type Vminobj: int
    :param Vminhole: if given the min volume threshold for the holes is not computed, Vminobj is used as the threshold 
    :type Vminhole: int
    :return: int numpy array, int, int
    """
    
    import numpy as np
    from skimage.measure import label
    from skimage.measure import regionprops
    
       
    image = (image > 0)*1
       
    if RemoveObjects:
        if Vminobj == None:
            regions_obj=regionprops(label(image))
            v_obj_beg=[]
            for i in range(len(regions_obj)):
                v_obj_beg.append(regions_obj[i].area)
            NumberOfObjects_beg = len(regions_obj)
            MaxVolObjects_beg = np.max(v_obj_beg)
            print(NumberOfObjects_beg,MaxVolObjects_beg)
            del(regions_obj)
            if len(v_obj_beg)>1:
                from skimage.morphology import remove_small_objects
                image = remove_small_objects(image, min_size=np.int(np.max(v_obj_beg)-2))
        #else:
        #    #Remove small objects
        #    from skimage.morphology import remove_small_objects
        #    image = remove_small_objects(label(image), min_size=Vminobj)
        #    if Verbose:
        #            print('Small object removed')
    
    if RemoveHoles:
        if Vminhole == None:
            image = (image < 1)*1
            
            regions_hol=regionprops(label(image))
            v_hol_beg=[]
            for i in range(len(regions_hol)):
                v_hol_beg.append(regions_hol[i].area)
            NumberOfHoles_beg = len(regions_hol)
            MaxVolHoles_beg = np.max(v_hol_beg)
            print(NumberOfHoles_beg,MaxVolHoles_beg)
            del(regions_hol)
            if len(v_hol_beg)>1:
                from skimage.morphology import remove_small_objects
                image = remove_small_objects(image, min_size=np.int(np.max(v_hol_beg)-2))
                image = (image < 1)*1
        #else:
        #    from skimage.morphology import remove_small_objects
        #    image = 1-remove_small_objects(label(1-image), min_size=Vminhole)
        #    if Verbose:    
        #        print('Small holes removed')
    
    #Bin closing
    if BinClosing:
        from skimage.morphology import closing
        from skimage.morphology import ball
        if closingradius == None:
            image = closing(image)
        else:
            image = closing(image, ball(ClosingRadius))
        if Verbose:
            print('Closing done')
    
    image = (image > 0)*1
    
    # If return the threshold volumes for objects and holes
    if GiveVolumes:
        return np.asarray(image, dtype='uint8'), np.int(np.max(v_obj_beg)-2), np.int(np.max(v_hol_beg)-2)
    
    return np.asarray(image, dtype='uint8')


def BubbleSegmentation(image, SigSeeds=1, SigWatershed=1, watershed_line=False, radius_opening=None, verbose=False, esti_min_dist=None, compactness=None):
    """
    Perform the bubble segmentation
    
    :param image: 3D image 
    :type image: int numpy array
    :param SigSeeds: Optional, Gaussian filter Sigma for the seeds
    :type SigSeeds: int
    :param SigWatershed: Optional, Gaussian filter Sigma for the watershed
    :type SigWatershed: int
    :param watershed_line: Optional, If True keep the 0-label surfaces between the segmented bubble regions
    :type watershed_line: Bool
    :param radius_opening: Optional, if not None, perform a radius opening operation on the labelled image with the given radius
    :type radius_opening: None or int
    :param verbose: Optional, if True, print progression steps of the segmentation
    :type verbose: Bool
    :return: int numpy array
    """
    
    import numpy as np
    from scipy import ndimage
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    from skimage.filters import gaussian
    from skimage.morphology import opening, ball
    
    # Distance map
    image = np.float16(image)
    Distmap = ndimage.distance_transform_edt(image)
    if verbose:
        print('Distance map: done')
    SmoothDistmap = gaussian(Distmap, sigma=SigSeeds);
    if verbose:
        print('Seeds distance map: done')
    
    # Extracting the seeds
    if esti_min_dist != None:
        local_max_coord = peak_local_max(SmoothDistmap, min_distance = int(esti_min_dist), exclude_border=False)
    else:
        local_max_coord = peak_local_max(SmoothDistmap, exclude_border=False)
    local_max_im = np.zeros(np.shape(image))
    for locmax in local_max_coord:
        local_max_im[locmax[0],locmax[1],locmax[2]] = 1
    del(local_max_coord)
    if verbose:
        print('Seeds: done')
    
    # Smooth distance map for watershed
    SmoothDistmap = gaussian(Distmap, sigma=SigWatershed)
    del(Distmap)
    if verbose:
        print('Watershed distance map: done')
    
    # Watershed
    if compactness != None:
        labelled_im = watershed(-SmoothDistmap, ndimage.label(local_max_im)[0], mask = image, compactness = compactness)
    else:
        labelled_im = watershed(-SmoothDistmap, ndimage.label(local_max_im)[0], mask = image)
    del(SmoothDistmap); del(local_max_im)
    if verbose:
        print('Watershed: done')
    
    #Opening
    if radius_opening!=None:
        labelled_im = opening(labelled_im, ball(radius_opening))
        if verbose:
            print('Opening: done')
            
    return labelled_im


def RemoveEdgeBubble(image, mask=None, rpercent=None, verbose=False, masktopbottom=None, returnmask=False):
    """
    Remove the bubbles on the image edges and in intersection with the mask (if given)
    
    :param image: 3D image 
    :type image: int numpy array
    :param image: 3D image, if given, removes also the labels at the intersection with the mask
    :type image: None or int numpy array
    :return: int numpy array
    """
    
    from spam.label.label import labelsOnEdges, removeLabels, makeLabelsSequential
    from skimage.measure import regionprops
    #from Package.Process.MaskCyl import MaskCyl
    import numpy as np
    
    if rpercent != None:
        if verbose:
            print('Radius given, a cylindrical mask was created')
        mask = MaskCyl(image, rpercent)
    if masktopbottom != None:
        mask[:masktopbottom[0]] = 0
        mask[masktopbottom[1]:] = 0
        if verbose:
            print('Crop top & bottom given, a top-bottom edge mask was created')
    
    # if mask
    if len(np.shape(mask))>0:
        # Masked labels
        imlabtouchmask = image*(1-mask)
        Reg = regionprops(imlabtouchmask)
        labtouchmask=[]
        for reg in Reg:
            labtouchmask.append(reg.label)
        # Remove masked labels
        image = removeLabels(image, labtouchmask)
        # Make sequential
        image = makeLabelsSequential(image)
        if verbose:
            print('Bubbles removed at the mask edges')
    
    # Remove top-bottom edge labels
    labedge = labelsOnEdges(image)
    image = removeLabels(image, labedge)
    image = makeLabelsSequential(image)
    if verbose:
        print('Bubbles removed at the top and bottom edges')
    
    if returnmask:
        return image, mask
    
    return image


#------------------------------------------------------------

def RemoveBackground_Batch(series, rawdir, prossdir, imrange, method='white_tophat', radius=5,  crop=None, bottom=None, verbose=False, Binning=None):
    from tifffile import imsave
    from FoamQuant.Helper import ReadRaw
    from FoamQuant.Helper import strindex
    #from Package.Basic.ReadRaw import ReadRaw
    #from Package.Process.RemoveBackground import RemoveBackground
    import spam.DIC
    
    for imi in imrange:
        image = ReadRaw(series, imi, rawdir, crop=crop)
        #image = RemoveBackground(image, method=method, radius=radius)
        
        # image string index
        imifordir = strindex(imi, n0)
        
        if Binning!=None:
            image = spam.DIC.deform.binning(image, Binning)
            imsave(prossdir + '/2_RemoveBackground/' + series + '/' + series+'_RemoveBackground_Bin'+str(Binning)+'_'+imifordir, image, bigtiff=True)
        else:
            imsave(prossdir + '/2_RemoveBackground/' + series + '/' + series+'_RemoveBackground_'+imifordir, image, bigtiff=True)
        
        if verbose:
            print(namesave+' '+str(imi)+': done')
            


def PhaseSegmentation_Batch(nameread, namesave, dirread, dirsave, imrange, method='ostu_global', th=None, radius=None, th0=None, th1=None, returnOtsu=False, verbose=False, ROIotsu=False, n0=3, endread='.tif',endsave='.tif'):
    from tifffile import imread, imsave
    import numpy as np
    from FoamQuant.Helper import strindex
    
    Lth=[]
    for imi in imrange:
        imifordir = strindex(imi, n0)
        # read image
        image = imread(dirread + nameread + imifordir + endread)
        
        # if keep otsu threshold
        if returnOtsu:
            # region of interest for Otsu
            if len(np.shape(ROIotsu))>0:
                print('ROIotsu', ROIotsu)
                image, th = PhaseSegmentation(image[ROIotsu[0]:ROIotsu[1],ROIotsu[2]:ROIotsu[3],ROIotsu[4]:ROIotsu[5]], 
                                              method='ostu_global',
                                              returnOtsu=returnOtsu)
            else:
                image, th = PhaseSegmentation(image, 
                                              method='ostu_global',
                                              returnOtsu=returnOtsu)
            Lth.append(th)
        else:
            image = PhaseSegmentation(image, method=method, th=th, radius=radius, th0=th0, th1=th1, returnOtsu=False)
            
        # save image
        imsave(dirsave + namesave + imifordir + endsave, np.asarray(image, dtype='uint8'), bigtiff=True)
        # if verbose
        if verbose:
            print(namesave+' '+str(imi)+': done')
    
    # return otsu threshold if true
    if returnOtsu:
        return Lth
    
def Masking_Batch(nameread, namesave, dirread, dirsave, imrange, verbose=False, sufix=None):
    from tifffile import imread, imsave
    #from Package.Process.MaskCyl import MaskCyl
    from FoamQuant.Helper import strindex
    
    for imi in imrange:
        # image string index
        imifordir = strindex(imi, n0)
        # read image
        if sufix!=None:
            image = imread(readdir + '/3_PhaseSegmented_'+sufix+'/' + series + '/' + series+'_PhaseSegmented_'+imifordir+'.tif')
        else:
            image = imread(readdir + '/3_PhaseSegmented/' + series + '/' + series+'_PhaseSegmented_'+imifordir+'.tif')
        
        if imi == imrange[0]:
            Mask = MaskCyl(image)    
        
        image = Mask*image
        
        # save image     
        if sufix!=None:
            imsave(savedir + '/4_Masked_'+sufix+'/' + series + '/' + series+'_Masked_'+imifordir+'.tif', image, bigtiff=True)
        else:
            imsave(savedir + '/4_Masked/' + series + '/' + series+'_Masked_'+imifordir+'.tif', image, bigtiff=True)
        
        # if verbose
        if verbose:
            print(namesave+imifordir+': done')
            
    
def RemoveSpeckleBin_Batch(nameread, namesave, dirread, dirsave, imrange, verbose=False, endread='.tif', endsave='.tif', n0=3, Cobj=0.5, Chole=0.5):
    from tifffile import imread, imsave
    #from Package.Process.RemoveSpeckleBin import RemoveSpeckleBin
    from FoamQuant.Helper import strindex
    
    # read first image
    imifordir = strindex(imrange[0], n0)
    image = imread(dirread + nameread + imifordir + endread)  
    image, Vobj, Vhole = RemoveSpeckleBin(image, GiveVolumes=True)
    print('First image (vox): maxObj',Vobj, 'maxHole',Vhole)
    # Volume thesholds
    Vminobj = round(Vobj*Cobj)
    Vminhole = round(Vhole*Chole)
    print('Thresholds (vox): thrObj',Vminobj, 'thrHole',Vminhole)
    
    for imi in imrange:
        imifordir = strindex(imi, n0)
        # read image
        image = imread(dirread + nameread + imifordir + endread)        
        # remove the small holes and objects
        image = RemoveSpeckleBin(image, Vminobj=Vminobj, Vminhole=Vminhole)
        # save image
        imsave(dirsave + namesave + imifordir + endsave, image, bigtiff=True)
        # if verbose
        if verbose:
            print(namesave+imifordir+': done')
            
            
def BubbleSegmentation_Batch(nameread, namesave, dirread, dirsave, imrange, verbose=False, endread='.tif', endsave='.tif', n0=3, writeparameters=False, Binning=None, esti_min_dist=None, SigSeeds=1, SigWatershed=1, watershed_line=False, compactness=None, radius_opening=None, ITK=False, ITKLevel=1):
    from tifffile import imread, imsave
    import os
    from FoamQuant.Helper import strindex
    
    #Check saving directory
    isExist = os.path.exists(dirsave)
    print('Path exist:', isExist)
    if not isExist:
        print('Error: saving path does not exist', dirsave)
        return
    
    if ITK:
        import spam.label
        labelled_im = spam.label.ITKwatershed.watershed(image, markers=None, watershedLevel=ITKLevel)
        if verbose:
            print(namesave+' '+str(imi)+': ITK done')
    else:
        # write parameters
        if writeparameters:
            file1 = open(dirsave + namesave + '_WatershedParameters.txt',"w")
            L = ["nameread \n",str(nameread),
                 "\n imrange \n",str(imrange),
                 "\n SigSeeds \n",str(SigSeeds),
                 "\n SigWatershed \n",str(SigWatershed),
                 "\n watershed_line \n",str(watershed_line),
                 "\n radius_opening \n",str(radius_opening),
                 "\n Twatershed_line \n",str(watershed_line),
                 "\n radius_opening \n",str(radius_opening),
                 "\n esti_min_dist \n",str(esti_min_dist),
                 "\n compactness \n",str(compactness)] 
            file1.writelines(L)
            file1.close() 

        for imi in imrange:
            # image string index
            imifordir = strindex(imi, n0)
            # read image
            image = imread(dirread + nameread + imifordir + endread)
            # if binning
            if Binning != None:
                import spam.DIC
                image = spam.DIC.deform.binning(image, Binning)

                Binning = SigSeeds//Binning
                SigWatershed = SigWatershed//Binning
                esti_min_dist = esti_min_dist//Binning
            # Bubble segmentation
            image = BubbleSegmentation(image, 
                                       SigSeeds, 
                                       SigWatershed, 
                                       watershed_line, 
                                       radius_opening, 
                                       verbose, 
                                       esti_min_dist, 
                                       compactness)
            # save image
            imsave(dirsave + namesave + imifordir + endsave, image, bigtiff=True)
            # if verbose
            if verbose:
                print(namesave+imifordir+': done')
            
            
def RemoveEdgeBubble_Batch(nameread, namesave, dirread, dirsave, imrange, verbose=False, endread='.tif', endsave='.tif', n0=3, maskcyl=False, masktopbottom=None, rpercent=None):
    from tifffile import imread, imsave
    #from Package.Process.RemoveEdgeBubble import RemoveEdgeBubble
    import os
    from FoamQuant.Helper import strindex
    
    #Check directory
    isExist = os.path.exists(dirread)
    print('Path exist:', isExist)
    if not isExist:
        print('Error: Saving path does not exist', dirread)
        return
        
    for imi in imrange:
        # image string index
        imifordir = strindex(imi, n0)
        # read image
        image = imread(dirread + nameread + imifordir + endread)
        
        # for the first image, get the cylindrical mask
        if maskcyl and imi == imrange[0]:
            image, mask = RemoveEdgeBubble(image, 
                                           masktopbottom=masktopbottom, 
                                           rpercent=rpercent, 
                                           returnmask=True) 
        elif maskcyl:
            image = RemoveEdgeBubble(image, 
                                     mask,
                                     returnmask=False)
        else:
            image = RemoveEdgeBubble(image,
                                     masktopbottom=masktopbottom,
                                     returnmask=False)
        # save image
        imsave(dirsave + namesave + imifordir + endsave, image, bigtiff=True)

        # if verbose
        if verbose:
            print(namesave+imifordir+': done')