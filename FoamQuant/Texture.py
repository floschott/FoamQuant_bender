def TextureTensor(image, image_noedge, IncludevonMises=False, maximumCoordinationNumber=30):
    import numpy as np
    from skimage.measure import regionprops
    import spam.label.contacts.labelledContacts as labelledContacts
    from Package.Quantify.Bubbles.InternalStain import InternalStain
    from Package.Quantify.Bubbles.vonMises_DiagVal import vonMises_DiagVal
    
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

    # Texture tensor M
    LM = []
    for h in range(1, len(contactTable)):
        z1,y1,x1 = centroid[h-1]
        M = np.zeros((3,3))
        count = 0
        
        for k in range(len(contactTable[h])//2):
            if contactTable[h][k*2] != 0:
                z2,y2,x2 = LCentroid[contactTable[h][k*2]-1]
                
                [Z,Y,X] = np.asarray([z2,y2,x2]) - np.asarray([z1,y1,x1])
                m = [[Z**2,Z*Y,Z*X],
                     [Y*Z,Y**2,Y*X],
                     [X*Z,X*Y,X**2]]
                M = M + m
                count = count + 1
                
        M = np.asarray(M)/count   # M = <m> = sum(m)/count
        LM.append(M)
            
    # Keep texture not at the edge
    LM_noegde = np.zeros((len(centroid_noedge),3,3))
    for h in range(len(LM)):
        z,y,x = centroid[h] 
        z = round(z); y = round(y); x = round(x)
        if image_noedge[z,y,x] != 0:  
            LM_noegde[image_noedge[z,y,x]-1] = LM[h]
    
    
    # Write for output
    Mzz = []; Myy = []; Mxx = []; Mzy = []; Mzx = []; Myx = []
    U1 = []; U2 = []; U3 = []
    VM = []
    
    for h in range(len(LM_noegde)):
        Mzz.append(LM_noedge[h][0][0])
        Myy.append(LM_noedge[h][1][1])
        Mxx.append(LM_noedge[h][2][2])
        Mzy.append(LM_noedge[h][0][1])
        Mzx.append(LM_noedge[h][0][2])
        Myx.append(LM_noedge[h][1][2])
        
        # If include von Mises
        if IncludevonMises:
            M = np.asarray([[Mzz[-1], Mzy[-1], Mzx[-1]], 
                            [Mzy[-1], Myy[-1], Myx[-1]], 
                            [Mzx[-1], Myx[-1], Mxx[-1]]])

            MVal,MVect = np.linalg.eig(M)
            u1,u2,u3 = InternalStrain(MVal1, MVal2, MVal3, dim=2)
            U1.append(u1)
            U2.append(u2)
            U3.append(u3)
            vM = vonMises_DiagVal(u1,u2,u3)
            VM.append(vM)

    return lab_noedge, centroid_noedge, Mzz,Myy,Mxx,Mzy,Mzx,Myx, U1,U2,U3, VM


def TextureTensor_BatchCSV(series, readdir, savedir, imrange, IncludevonMises=False, verbose=False):
    import numpy as np
    from tifffile import imread
    import csv
    
    from Package.Quantify.Bubbles.TextureTensor import TextureTensor
    
    #Check save directory
    path = savedir + '/3_Texture/' + series
    isExist = os.path.exists(path)
    print('Path exist:', isExist)
    if not isExist:
        return
    
    #Batch loop
    for imi in imrange:
        # image string index
        imistr = str(imi)
        imistrlen = len(imistr)
        imifordir = (3-imistrlen)*'0'+imistr
        
        # read image
        image = imread(readdir + '/5_BubbleSegmented/' + series + '/' + series+'_BubbleSegmented_'+imifordir+'.tif')
        image_noegde = imread(readdir + '/6_BubbleSegmented_NoEdge/' + series + '/' + series+'_BubbleSegmented_NoEdge_'+imifordir+'.tif')
        
        # ShapeTensor data
        PropList = TextureTensor(image, image_noegde, IncludevonMises=IncludevonMises)  
        
        # Save in TSV
        with open(path+'/'+series+'_Texture_'+imi+'.tsv', 'w', newline='') as csvfile:        
            writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            if IncludevonMises:
                writer.writerow(['Label','z','y','x','Mzz','Myy','Mxx','Mzy','Mzx','Myx', 'U1','U2','U3', 'VM'])
                for i in range(len(PropList[0])):
                    writer.writerow([PropList[0][i],
                                     PropList[1][i][0],PropList[1][i][1],PropList[1][i][2],
                                     PropList[2][i],PropList[3][i],PropList[4][i],PropList[5][i],PropList[6][i],PropList[7][i]],
                                     PropList[8][i],PropList[9][i],PropList[10][i],
                                     PropList[11][i])
            else:
                writer.writerow(['Label','z','y','x','Mzz','Myy','Mxx','Mzy','Mzx','Myx'])
                for i in range(len(PropList[0])):
                    writer.writerow([PropList[0][i],
                                     PropList[1][i][0],PropList[1][i][1],PropList[1][i][2],
                                     PropList[2][i],PropList[3][i],PropList[4][i],PropList[5][i],PropList[6][i],PropList[7][i]])
        if verbose:
            print('Image'+imifordir+' Nregions: ',len(PropList[0]), ':done')
            
def CylTavg_forced(Coord_Cyl, T_Cyl, Range=[0,1000,0,600], N=[5,5], verbose=True, CountMin=0, CstVolRange=False):
    import numpy as np
    
    if CstVolRange:
        LEps_1=[]
        LGrid1=[]
        for i in range(N[0]):
            if i == 0:
                beg = 0
                end = Range[1]/np.sqrt(N[0])
                mid = (beg+end)/2
                eps = (end-beg)/2
                LEps_1.append(eps)
                LGrid1.append(mid)
            else:
                beg = end
                end = np.sqrt(np.power(Range[1],2)/N[0]+np.power(beg,2))
                mid = (beg+end)/2
                eps = (end-beg)/2
                LEps_1.append(eps)
                LGrid1.append(mid)

    else:
        Eps_1 = (Range[1]-Range[0])/(2*N[0])*1.0
        LEps_1 = [Eps_1]*N[0]
        LGrid1 = np.linspace(Range[0]+Eps_1, Range[1]-Eps_1, N[0])
        
        
    Eps_2 = (Range[3]-Range[2])/(2*N[1])*1.0
    LEps_2 = [Eps_2]*N[1]
    LGrid2 = np.linspace(Range[2]+Eps_2, Range[3]-Eps_2, N[1])
    
   
    
       
    if verbose == True:
        print('Grid:\n', LGrid1,'\n',LGrid2)
    
    Coord1_avg=np.full([N[0], N[1]], np.nan)
    Coord2_avg=np.full([N[0], N[1]], np.nan)
    Count=np.full([N[0], N[1]], np.nan)
    Reff=np.full([N[0], N[1]], np.nan)
    Valmin=np.full([N[0], N[1]], np.nan)
    Valmax=np.full([N[0], N[1]], np.nan)
    Vectminr=np.full([N[0], N[1]], np.nan)
    Vectmaxr=np.full([N[0], N[1]], np.nan)
    Vectminz=np.full([N[0], N[1]], np.nan)
    Vectmaxz=np.full([N[0], N[1]], np.nan)
    Valphi=np.full([N[0], N[1]], np.nan)
    VM_before=np.full([N[0], N[1]], np.nan)
    VM_after=np.full([N[0], N[1]], np.nan)
    
    for i1 in range(N[0]):
        for i2 in range(N[1]):

            count = 0
            Srz22 = np.zeros([2,2])
            Sphi = 0
            
            for i_obj in range(len(T_Cyl)):
                if Coord_Cyl[i_obj][0] > LGrid1[i1] - LEps_1[i1] and Coord_Cyl[i_obj][0] < LGrid1[i1] + LEps_1[i1]:
                    if Coord_Cyl[i_obj][2] > LGrid2[i2] - LEps_2[i2] and Coord_Cyl[i_obj][2] < LGrid2[i2] + LEps_2[i2]:
                                
                            count += 1                                         # count
                            Srz22[0][0] = Srz22[0][0] + T_Cyl[i_obj][0][0]     # sum of the values
                            Srz22[0][1] = Srz22[0][1] + T_Cyl[i_obj][0][2]     # sum of the values
                            Srz22[1][0] = Srz22[1][0] + T_Cyl[i_obj][2][0]     # sum of the values
                            Srz22[1][1] = Srz22[1][1] + T_Cyl[i_obj][2][2]     # sum of the values
                            Sphi += T_Cyl[i_obj][1][1]                      # sum of the values
            
            if count > CountMin:
                Coord1_avg[i1, i2] = LGrid1[i1]
                Coord2_avg[i1, i2] = LGrid2[i2]
                Count[i1, i2] = count
                Valphi[i1, i2] = np.asarray(Sphi)/count
                
                # S+, S-
                Srz22 = np.asarray(Srz22)/count          
                EVal, EVect = np.linalg.eig(Srz22)
                iValmax = np.argmax(np.abs(EVal))
                
                # Reff
                reff = np.power(EVal[0]*EVal[1]*np.asarray(Sphi)/count, 1/3)
                Reff[i1, i2] = reff
                
                # S+, S- and VectS+, VectS-
                Valmax[i1, i2] = EVal[iValmax]
                Valmin[i1, i2] = EVal[1-iValmax]
                
                Vectmaxr[i1, i2] = EVect[0][iValmax]
                Vectmaxz[i1, i2] = EVect[1][iValmax]
                Vectminr[i1, i2] = EVect[0][1-iValmax]
                Vectminz[i1, i2] = EVect[1][1-iValmax]
    
   
    # Def Szr
    DefValmax = np.log(Valmax/Reff)
    DefValmin = np.log(Valmin/Reff)
    DefValphi = np.log(Valphi/Reff)
                
    return [LGrid1,LGrid2], [Coord1_avg, Coord2_avg], Count,Reff, [Valmax,Valmin,Valphi], [Vectmaxr,Vectmaxz,Vectminr,Vectminz], [DefValmax,DefValmin,DefValphi]

            