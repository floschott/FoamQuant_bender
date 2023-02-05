def RegionProp(image, field=False):
    """
    Return basic region properties from a labeled image: labels [0], centroids [1], volumes [2], (inertia components [3])
    
    :param image: 3D image 
    :type image: int numpy array
    :param IncludeInertia: if True, also return inertial components
    :type IncludeInertia: Bool
    :return: array of labels [0], centroids [1], volumes [2], (inertia components [3])
    """
    import numpy as np
    from skimage.measure import regionprops
    
    if not field:
        field = [0,np.inf,0,np.inf,0,np.inf]
    Reg = regionprops(image)
    
    lab = []; centroid = []; vol = []; rad = []
    area = []; sph = []; volT =[]
    
    La=[];Lb=[];Lc=[] 
    Laz=[];Lay=[];Lax=[]
    Lbz=[];Lby=[];Lbx=[]
    Lcz=[];Lcy=[];Lcx=[]
    
    LUa=[];LUb=[];LUc=[]; LU=[]; Ltype=[]
    
    for reg in Reg:
        z,y,x = reg.centroid
        if z>field[0] and z<field[1] and y>field[2] and y<field[3] and x>field[4] and x<field[5]:
            lab.append(reg.label)
            centroid.append([z,y,x])
            V = reg.area
            vol.append(V)
            req=np.power(V*3/(4*np.pi), 1/3)
            rad.append(req)

            eig1,eig2,eig3 = reg.inertia_tensor_eigvals
            I = reg.inertia_tensor
            Val, Vect = np.linalg.eig(I)
            
            oVect = [];eig=[eig1,eig2,eig3]
            for eigi in range(3):
                for val in Val:
                    if round(val) == round(eig[eigi]):
                        oVect.append(Vect[eigi])
            
            if eig2+eig3-eig1>0 and eig1+eig3-eig2>0 and eig1+eig2-eig3>0:
                # Semi-axes
                a = np.sqrt(5/2*(eig2+eig3-eig1))
                b = np.sqrt(5/2*(eig1+eig3-eig2))
                c = np.sqrt(5/2*(eig1+eig2-eig3))
                # Ordered semi-axes
                sa,sb,sc = np.sort([a,b,c])
                La.append(sa)
                Lb.append(sb)
                Lc.append(sc)
                # Ordered corresponding eig-vectors
                ooVect=[]; eig=[a,b,c]
                for si in [sa,sb,sc]:
                    for eigi in range(3):
                        if si == eig[eigi]:
                            ooVect.append(oVect[eigi])
                oVect=ooVect
                a=sa; b=sb; c=sc
                Laz.append(oVect[0][0]); Lay.append(oVect[0][1]); Lax.append(oVect[0][2])
                Lbz.append(oVect[1][0]); Lby.append(oVect[1][1]); Lbx.append(oVect[1][2])
                Lcz.append(oVect[2][0]); Lcy.append(oVect[2][1]); Lcx.append(oVect[2][2])
                
                # Volume from axes
                V = 4/3*np.pi*a*b*c
                volT.append(V)
                p = 1.6
                pa = np.power(a,p); pb = np.power(b,p); pc = np.power(c,p)
                # Area
                A = 4*np.pi*np.power((pa*pb+pa*pc+pb*pc)/3,1/p)
                area.append(A)
                # Sphericity
                sph.append(np.power(np.pi,1/3)*np.power(6*V,2/3)/A)
                # Strain
                Ua = np.log(a/req)
                Ub = np.log(b/req)
                Uc = np.log(c/req)
                LUa.append(Ua)
                LUb.append(Ub)
                LUc.append(Uc)
                # Strain vM invariant
                LU.append(np.sqrt(0.5*(np.power(Ua-Ub, 2)+np.power(Ua-Uc,2)+np.power(Ub-Uc,2))))
                # Type: oblate or prolate
                if np.abs(Ua) > np.abs(Uc):
                    Ltype.append(1)
                else:
                    Ltype.append(-1)
                
            else:
                La.append(-1)
                Lb.append(-1)
                Lc.append(-1)
                
                Laz.append(-1)
                Lay.append(-1)
                Lax.append(-1)
                
                Lbz.append(-1)
                Lby.append(-1)
                Lbx.append(-1)
                
                Lcz.append(-1)
                Lcy.append(-1)
                Lcx.append(-1)
                volT.append(-1)
                area.append(-1)     
                sph.append(-1)
                LUa.append(-1)
                LUb.append(-1)
                LUc.append(-1)
                LU.append(-1)
                Ltype.append(-1)

                print('Error: negative shape value, coord',centroid[-1], 'vol', vol)
        
    return lab,centroid,vol, rad,area,sph,volT, La,Lb,Lc, Laz,Lay,Lax, Lbz,Lby,Lbx, Lcz,Lcy,Lcx, LUa,LUb,LUc,LU,Ltype
    
def RegionProp_Batch(nameread, namesave, dirread, dirsave, imrange, verbose=False, endread='.tiff', endsave='.tsv', n0=3,field=False):
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
    from FoamQuant.FromLabelled import RegionProp
    from FoamQuant.Helper import strindex
    import os
    
    #Check directory
    isExist = os.path.exists(dirsave)
    print('Path exist:', isExist)
    if not isExist:
        print('Saving path does not exist:\n', isExist)
        return
    
    #Batch loop
    for imi in imrange:
        # image string index
        imifordir = strindex(imi, n0)
        # read image
        image = imread(dirread + nameread + imifordir + endread)
        Prop = RegionProp(image, field=field)
        # Save as TSV
        with open(dirsave + namesave + imifordir + endsave, 'w', newline='') as csvfile:        
            writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['lab','z','y','x', 'vol','rad','area','sph','volfit', 'S1','S2','S3','e1z','e1y','e1x','e2z','e2y','e2x','e3z','e3y','e3x','U1','U2','U3','U','type'])
            for i in range(len(Prop[0])):
                writer.writerow([Prop[0][i], Prop[1][i][0],Prop[1][i][1],Prop[1][i][2], 
                                 Prop[2][i],Prop[3][i], Prop[4][i],Prop[5][i],Prop[6][i],
                                 Prop[7][i],Prop[8][i], Prop[9][i],
                                 Prop[10][i],Prop[11][i], Prop[12][i],
                                 Prop[13][i],Prop[14][i], Prop[15][i],
                                 Prop[16][i],Prop[17][i], Prop[18][i],
                                 Prop[19][i],Prop[20][i], Prop[21][i],Prop[22][i],Prop[23][i]])
        if verbose:
            print(namesave+imifordir+': done')

            
def Read_RegionProp(nameread, dirread, imrange, verbose=False, endread='.tsv', n0=3):            
    import numpy as np 
    import pandas as pd
    from FoamQuant.Helper import strindex
    import csv
    
    lab=[]; z=[];y=[];x=[]; vol=[]; rad=[]
    area=[]; sph=[]; volT=[]
    La=[];Lb=[];Lc=[] 
    Laz=[];Lay=[];Lax=[]
    Lbz=[];Lby=[];Lbx=[]
    Lcz=[];Lcy=[];Lcx=[]
    LUa=[];LUb=[];LUc=[]; LU=[]; Ltype=[]
    
    #Batch loop
    for imi in imrange:
        # image string index
        imifordir = strindex(imi, n0)

        Regprops = pd.read_csv(dirread+nameread+imifordir+endread, sep='\t',engine="python",  quoting=csv.QUOTE_NONE)
        lab.append(np.asarray(Regprops['lab']))
        z.append(np.asarray(Regprops['z']))
        y.append(np.asarray(Regprops['y']))
        x.append(np.asarray(Regprops['x']))
        vol.append(np.asarray(Regprops['vol']))
        rad.append(np.asarray(Regprops['rad']))
        area.append(np.asarray(Regprops['area']))
        sph.append(np.asarray(Regprops['sph']))
        volT.append(np.asarray(Regprops['volfit']))
        La.append(np.asarray(Regprops['S1']))
        Lb.append(np.asarray(Regprops['S2']))
        Lc.append(np.asarray(Regprops['S3'])) 
            
        Laz.append(np.asarray(Regprops['e1z']))
        Lay.append(np.asarray(Regprops['e1y']))
        Lax.append(np.asarray(Regprops['e1x']))
        Lbz.append(np.asarray(Regprops['e2z']))
        Lby.append(np.asarray(Regprops['e2y']))
        Lbx.append(np.asarray(Regprops['e2x']))
        Lcz.append(np.asarray(Regprops['e3z']))
        Lcy.append(np.asarray(Regprops['e3y']))
        Lcx.append(np.asarray(Regprops['e3x']))
        
        LUa.append(np.asarray(Regprops['U1']))
        LUb.append(np.asarray(Regprops['U2']))
        LUc.append(np.asarray(Regprops['U3']))
        LU.append(np.asarray(Regprops['U']))
        Ltype.append(np.asarray(Regprops['type']))
        
        if verbose:
                print(namesave+imifordir+': done')
        
    lab=np.concatenate(lab)
    z=np.concatenate(z)
    y=np.concatenate(y)
    x=np.concatenate(x)
    vol=np.concatenate(vol)
    rad=np.concatenate(rad)
    area=np.concatenate(area)
    sph=np.concatenate(sph)
    volT=np.concatenate(volT)
    La=np.concatenate(La)
    Lb=np.concatenate(Lb)
    Lc=np.concatenate(Lc)
    Laz=np.concatenate(Laz)
    Lay=np.concatenate(Lay)
    Lax=np.concatenate(Lax)
    Lbz=np.concatenate(Lbz)
    Lby=np.concatenate(Lby)
    Lbx=np.concatenate(Lbx)
    Lcz=np.concatenate(Lcz)
    Lcy=np.concatenate(Lcy)
    Lcx=np.concatenate(Lcx)
    LUa=np.concatenate(LUa)
    LUb=np.concatenate(LUb)
    LUc=np.concatenate(LUc)
    LU=np.concatenate(LU)
    Ltype=np.concatenate(Ltype)
    
    Properties={'lab':lab,'z':z,'y':y,'x':x,'vol':vol,'rad':rad,'area':area,'sph':sph,'volfit':volT,'S1':La,'S2':Lb,'S3':Lc,
                'e1z':Laz,'e1y':Lay,'e1x':Lax,'e2z':Lbz,'e2y':Lby,'e2x':Lbx,'e3z':Lcz,'e3y':Lcy,'e3x':Lcx,
                'U1':LUa,'U2':LUb,'U3':LUc,'U':LU,'type':Ltype}
    return Properties
        

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


def Coordination_Batch(readdir, readdir_noedge, dirsave, imrange, readend='.tiff', readend_noedge='.tiff', maximumCoordinationNumber=30, verbose=False, n0=3):
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
    from FoamQuant.Helper import strindex
    import os

    #Check directory
    isExist = os.path.exists(dirsave)
    print('Path exist:', isExist)
    if not isExist:
        print('Saving path does not exist:\n', isExist)
        return
    
    #Batch loop
    for imi in imrange:
        # image string index
        imifordir = strindex(imi, n0)
        
        # read image
        image = imread(readdir+imistr+readend)
        image_noegde = imread(readdir_noedge+imistr+readend_noedge)
        
        # ShapeTensor data
        Prop = Coordination(image, 
                            image_noegde,
                            maximumCoordinationNumber=maximumCoordinationNumber, 
                            returnCoordinationImage=False)
        
        # Save as TSV
        with open(dirsave+'/Coordination_'+imi+'.tsv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Label','z','y','x','Z'])
            for i in range(len(Prop[0])):
                writer.writerow([Prop[0][i], Prop[1][i][0],Prop[1][i][1],Prop[1][i][2], Prop[2][i]])
            
        if verbose:
            print('Image'+imifordir+' Nregions: ',len(PropList[0]), ':done')
            
            

            
