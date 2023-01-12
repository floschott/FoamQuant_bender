def RegionProp(image):
    """
    Return basic region properties from a labeled image: labels [0], centroids [1], volumes [2], (inertia components [3])
    
    :param image: 3D image 
    :type image: int numpy array
    :param IncludeInertia: if True, also return inertial components
    :type IncludeInertia: Bool
    :return: array of labels [0], centroids [1], volumes [2], (inertia components [3])
    """
    
    from skimage.measure import regionprops
    Reg = regionprops(image)
    lab = []; centroid = []; vol = []; rad = []; sph = []; volT =[]
    for reg in Reg:
        lab.append(reg.label)
        centroid.append(reg.centroid)
        V = reg.area
        vol.append(V)
        req=np.power(V*3/(4*np.pi), 1/3)
        rad.append(Req)
            
        eig1,eig2,eig3 = reg.inertia_tensor_eigvals
        a = np.sqrt((eig2+eig3-eig1)/2)
        b = np.sqrt((eig1+eig3-eig2)/2)
        c = np.sqrt((eig1+eig2-eig3)/2)
        V = 4/3*np.pi*a*b*c
        volT.append(V)
        p = 1.6
        pa = np.power(a,p)
        pb = np.power(b,p)
        pc = np.power(c,p)
        A = 4*np.pi*np.power(pa*pb+pa*pc+pb*pc,1/p)
        sph.append(np.power(np.pi,1/3)*np.power(6*V,2/3)/A)
        
    return lab,centroid,vol,sph,volT

    
def RegionProp_Batch(readdir, savedir, imrange, readend='.tif', IncludeInertia=False, verbose=False):
    """
    Run RegionProp function on a batch of images and save the outputs as .tsv
    
    :param readdir: Labeled images folder
    :type readdir: str
    :param readdir: folder to save the .tsv doc
    :type readdir: str
    :param readend: tiff image saving end, default is '.tiff'
    :type readend: str
    :param imrange: list of image indexes
    :type imrange: int array
    :param IncludeInertia: if True, also return inertial components
    :type IncludeInertia: Bool
    :param verbose: if True, print verbose including the number of labels
    :type verbose: Bool
    """   
    
    import numpy as np
    from tifffile import imread
    import csv
    import RegionProp
    from Package.BasicTools import strindex
    
    #Check directory
    isExist = os.path.exists(savedir)
    print('Path exist:', isExist)
    if not isExist:
        print('Saving path does not exist:\n', isExist)
        return
    
    #Batch loop
    for imi in imrange:
        # image string index
        imistr= strindex(imi,3)
        # read image
        image = imread(readdir+imistr+readend)
        Prop = RegionProp(image, IncludeInertia=IncludeInertia)
        # Save as TSV
        with open(savedir+'/RegionProp_'+imistr+'.tsv', 'w', newline='') as csvfile:        
            writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Label','z','y','x','Volume','Radius','Sphericity','VolumeT'])
            for i in range(len(Prop[0])):
                writer.writerow([Prop[0][i], Prop[1][i][0],Prop[1][i][1],Prop[1][i][2], Prop[2][i],Prop[3][i]])
        if verbose:
            print('Image '+imistr+', Nreg: ',len(Prop[0]), ':done')

            

def Coordination(image, image_noedge, maximumCoordinationNumber=30, returnCoordinationImage=False):
    """
    Return labels [0], centroids [1], coordinations [2] of the no-edge image, (and coordination image [3])
    
    :param image: full 3D image 
    :type image: int numpy array
    :param image_noedge: 3D image with removed label at the edges
    :type image_noedge: int numpy array
    :param maximumCoordinationNumber: the maximum number of coordination, default 30
    :type maximumCoordinationNumber: int
    :param returnCoordinationImage: if True, additionally returns image_noedge coordination image
    :type returnCoordinationImage: Bool
    :return: labels [0], centroids [1], coordinations [2] of the no-edge image, (and coordination image [3])
    """    
    
    import numpy as np
    import spam.label.contacts.labelledContacts as labelledContacts
    
    # Regions Im edge
    reg = regionprops(image)
    centroid = []
    for reg in reg:
        centroid.append(reg.centroid)
    # Regions Im noedge
    reg_noedge = regionprops(image_noedge)
    centroid_noedge = []; lab_noedge=[]
    for reg in reg_noedge:
        lab_noedge.append(reg.centroid)
        centroid_noedge.append(reg.centroid)

    # Contacts Im edge
    contactVolume, Z, contactTable, contactingLabels = labelledContacts(image)
    del(contactVolume)
                
    # Keep Coordination at the edge
    Z_noegde = np.zeros(len(centroid_noedge))
    for h in range(len(Z)):
        z,y,x = centroid[h]                       
        z = round(z); y = round(y); x = round(x)
        if image_noedge[z,y,x] != 0: 
            Z_noegde[image_noedge[z,y,x]-1] = Z[h]
            
    if returnCoordImage:
        CoordinationImage = spam.label.label.convertLabelToFloat(image_noedge, Z_noegde)
        return lab_noedge, centroid_noedge, Z, CoordinationImage
    return lab_noedge, centroid_noedge, Z


def Coordination_Batch(readdir, readdir_noedge, savedir, imrange, readend='.tif', readend_noedge='.tif', maximumCoordinationNumber=30, verbose=False):
    """
    Return labels [0], centroids [1], coordinations [2] of the no-edge image, (and coordination image [3])
    
    :param image: full 3D image 
    :type image: int numpy array
    :param image_noedge: 3D image with removed label at the edges
    :type image_noedge: int numpy array
    :param maximumCoordinationNumber: the maximum number of coordination, default 30
    :type maximumCoordinationNumber: int
    :param returnCoordinationImage: if True, additionally returns image_noedge coordination image
    :type returnCoordinationImage: Bool
    :return: labels [0], centroids [1], coordinations [2] of the no-edge image, (and coordination image [3])
    """     
    
    import numpy as np
    from tifffile import imread, imsave
    import csv
    import Coordination
    from Package.BasicTools import strindex

    #Check directory
    isExist = os.path.exists(savedir)
    print('Path exist:', isExist)
    if not isExist:
        print('Saving path does not exist:\n', isExist)
        return
    
    #Batch loop
    for imi in imrange:
        # image string index
        imistr= strindex(imi,3)
        
        # read image
        image = imread(readdir+imistr+readend)
        image_noegde = imread(readdir_noedge+imistr+readend_noedge)
        
        # ShapeTensor data
        Prop = Coordination(image, 
                            image_noegde,
                            maximumCoordinationNumber=maximumCoordinationNumber, 
                            returnCoordinationImage=False)
        
        # Save as TSV
        with open(savedir+'/Coordination_'+imi+'.tsv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Label','z','y','x','Z'])
            for i in range(len(Prop[0])):
                writer.writerow([Prop[0][i], Prop[1][i][0],Prop[1][i][1],Prop[1][i][2], Prop[2][i]])
            
        if verbose:
            print('Image'+imifordir+' Nregions: ',len(PropList[0]), ':done')
            
            

            