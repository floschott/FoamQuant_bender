def InternalStrain(Val1, Val2, Val3, dim=1):
    import numpy as np
    
    Req = np.power(Val1*Val2*Val3, 1/3)
    
    U1 = np.log(Val1/Req)/dim
    U2 = np.log(Val2/Req)/dim
    U3 = np.log(Val3/Req)/dim
    
    return U1, U2, U3

def vonMises_DiagVal(U1, U2, U3):
    import numpy as np
    return np.sqrt(0.5*(np.power(U1-U2, 2)+np.power(U1-U3,2)+np.power(U2-U3,2)))

def ShapeForVTK(Coord, TS, Count):
    import numpy as np
    from Package.Quantify.Strain.InternalStrain import InternalStrain
    from Package.Quantify.Strain.vonMises_DiagVal import vonMises_DiagVal
    
    TUS=[]
    OblateProlate=[]
    VM=[]

    for i in range(len(Coord)):    
        if Count[i] > 0:

            Val, Vect = np.linalg.eig(TS[i])
            u1,u2,u3=InternalStrain(Val[0], Val[1], Val[2], dim=1)
            uS = np.asarray([[u1, 0, 0],
                            [0, u2, 0],
                            [0, 0, u3]])
            P0zyx = np.transpose(Vect)
            uS = np.transpose(P0zyx)@uS@P0zyx
            TUS.append(uS)

            USmax = np.max([u1, u2, u3])
            USmin = np.min([u1, u2, u3])
            if np.abs(USmax) > np.abs(USmin):
                OblateProlate.append(1)
            else:
                OblateProlate.append(-1)


            vM=vonMises_DiagVal(u1, u2, u3)
            VM.append(vM)

    TUS=np.asarray(TUS)
    OblateProlate=np.asarray(OblateProlate)
    VM=np.asarray(VM)
    
    return TUS, OblateProlate, VM


def ReadStrainCSV(readdir,series,imrange,verbose=False):
    import numpy as np
    import csv
    
    LData=[]
    for imi in imrange:
        Data=[]
        # image string index
        imistr = str(imi)
        imistrlen = len(imistr)
        imifordir = (3-imistrlen)*'0'+imistr
        if verbose:
            print(imifordir)

        with open(readdir+ '/4_Shape/' + series+'/'+series+'_Shape_'+imifordir+'.tsv', 'r', newline='') as csvfile:        
                reader = csv.reader(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                rowi=0
                for row in reader:
                    if rowi>-1:
                        Data.append(row)
                    rowi+=1

        Data = np.asarray(Data)
        LData.append(Data)

    if verbose:
        print('Data read')
        
    DataPack = []
    for j in range(len(LData)):
        Lab=[]
        Coord=[]
        TS=[]; TUS=[]
        OblateProlate=[]
        VM=[]
        
        if verbose:
            # image string index
            imistr = str(imrange[j])
            imistrlen = len(imistr)
            imifordir = (3-imistrlen)*'0'+imistr
            print(imifordir)
            
        for i in range(1,len(LData[j])):
            Data=LData[j]
            lab = np.asarray(Data[i][0], dtype='int')
            
            z = np.asarray(Data[i][1], dtype='float')
            y = np.asarray(Data[i][2], dtype='float')
            x = np.asarray(Data[i][3], dtype='float')

            Szz = np.asarray(Data[i][16], dtype='float')
            Syy = np.asarray(Data[i][17], dtype='float')
            Sxx = np.asarray(Data[i][18], dtype='float')
            Szy = np.asarray(Data[i][19], dtype='float')
            Szx = np.asarray(Data[i][20], dtype='float')
            Syx = np.asarray(Data[i][21], dtype='float')

            S = [[Szz,Szy,Szx],[Szy,Syy,Syx],[Szx,Syx,Sxx]]

            US1 = np.asarray(Data[i][22], dtype='float')
            US2 = np.asarray(Data[i][23], dtype='float')
            US3 = np.asarray(Data[i][24], dtype='float')

            USzz = np.asarray(Data[i][25], dtype='float')
            USyy = np.asarray(Data[i][26], dtype='float')
            USxx = np.asarray(Data[i][27], dtype='float')
            USzy = np.asarray(Data[i][28], dtype='float')
            USzx = np.asarray(Data[i][29], dtype='float')
            USyx = np.asarray(Data[i][30], dtype='float')

            vM = np.asarray(Data[i][31], dtype='float')

            US = [[USzz,USzy,USzx],[USzy,USyy,USyx],[USzx,USyx,USxx]]

            if np.isnan(z)+np.isnan(y)+np.isnan(x)+np.isnan(Szz)+np.isnan(Syy)+np.isnan(Sxx)+np.isnan(Szy)+np.isnan(Szx)+np.isnan(Syx)+np.isnan(vM) < 1:

                USmax = np.max([US1, US2, US3])
                USmin = np.min([US1, US2, US3])
                if np.abs(USmax) > np.abs(USmin):
                    OblateProlate.append(1)
                else:
                    OblateProlate.append(-1)

                Coord.append([z,y,x])
                TS.append(S)
                TUS.append(US)
                VM.append(vM)
                Lab.append(lab)

        Coord=np.asarray(Coord)
        TS=np.asarray(TS)
        TUS=np.asarray(TUS)
        OblateProlate=np.asarray(OblateProlate)
        VM=np.asarray(VM)
    
        DataPack.append({"imi":imi, "Label":Lab, "Coord":Coord,"TS":TS,"TUS":TUS,"OblateProlate":OblateProlate,"VM":VM})
    
    if verbose:
        print('Data in arrays')
        
    return DataPack


def ShapeTensor(image, IncludeStrain=False):
    import numpy as np
    from skimage.measure import regionprops
    from Package.Quantify.Strain.InternalStrain import InternalStrain
    from Package.Quantify.Strain.vonMises_DiagVal import vonMises_DiagVal
    
    
    Reg = regionprops(image)

    lab = []; centroid = []
    Vect1 = []; Vect2 = []; Vect3 = []
    
    S1 = []; S2 = []; S3 = []
    Szz = []; Syy = []; Sxx = []; Szy = []; Szx = []; Syx = []
    
    U1 = []; U2 = []; U3 = []
    USzz = []; USyy = []; USxx = []; USzy = []; USzx = []; USyx = []
    VM = []
    
    for j in range(len(Reg)):
        lab.append(Reg[j].label)
        centroid.append(Reg[j].centroid)
        vol=Reg[j].area
            
        I = Reg[j].inertia_tensor
        Izz=I[0][0]
        Iyy=I[1][1]
        Ixx=I[2][2]
        Izy=I[0][1]
        Izx=I[0][2]
        Iyx=I[1][2]
        
        # Inertia tensor preparation for Shape tensor
        Jzz=(Iyy+Ixx-Izz)/(2*vol)
        Jyy=(Iyy+Ixx-Izz)/(2*vol)
        Jxx=(Iyy+Ixx-Izz)/(2*vol)
        Jzy=-Izy/vol
        Jzx=-Izx/vol
        Jyx=-Iyx/vol
        J = np.asarray([[Jzz, Jzy, Jzx], 
                        [Jzy, Jyy, Jyx], 
                        [Jzx, Jyx, Jxx]])
        
        # S = sqrt(J)
        JVal, JVect = np.linalg.eig(J)
        
        JVect
        
        if JVal[0]<0 or JVal[0]<0 or JVal[0]<0:
            print(J, JVal)
            
        Vect1.append(JVect[0]); Vect2.append(JVect[1]); Vect3.append(JVect[2])    
            
        [SVal1, SVal2, SVal3] = np.sqrt(JVal)
        S1.append(SVal1);S2.append(SVal2); S3.append(SVal3)
        
        S = np.asarray([[SVal1, 0, 0],
                        [0, SVal2, 0],
                        [0, 0, SVal3]])
        P0zyx = np.transpose(JVect)
        S = np.transpose(P0zyx)@S@P0zyx
        Szz.append(S[0][0])
        Syy.append(S[1][1])
        Sxx.append(S[2][2])
        Szy.append(S[0][1])
        Szx.append(S[0][2])
        Syx.append(S[1][2])
        
        # If include von Mises
        if IncludeStrain:
            # Strain eigenvalues
            u1,u2,u3=InternalStrain(SVal1, SVal2, SVal3, dim=1)
            U1.append(u1)
            U2.append(u2)
            U3.append(u3)
            
            # von Mises
            vM=vonMises_DiagVal(u1, u2, u3)
            VM.append(vM)
            
            # Strain tensor
            US = np.asarray([[u1, 0, 0],
                            [0, u2, 0],
                            [0, 0, u3]])
            US = np.transpose(P0zyx)@US@P0zyx
            USzz.append(US[0][0])
            USyy.append(US[1][1])
            USxx.append(US[2][2])
            USzy.append(US[0][1])
            USzx.append(US[0][2])
            USyx.append(US[1][2])
    
    if IncludeStrain:
        return lab, centroid, Vect1,Vect2,Vect3, S1,S2,S3, Szz,Syy,Sxx,Szy,Szx,Syx, U1,U2,U3, USzz,USyy,USxx,USzy,USzx,USyx, VM
            
    return lab, centroid, Vect1,Vect2,Vect3, S1,S2,S3, Szz,Syy,Sxx,Szy,Szx,Syx
            
            
def ShapeTensor_BatchCSV(series, readdir, savedir, imrange, IncludeStrain=False, verbose=False, sufix=None, binvalue=None):
    import numpy as np
    from tifffile import imread
    import csv
    import os
    
    from Package.Quantify.Strain.ShapeTensor import ShapeTensor
    
    #Check save directory
    path = savedir + '/4_Shape/' + series
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
        if binvalue!=None and sufix != None:
            image = imread(readdir + '/7_BubbleSegmented_NoEdge_'+sufix+'/' + series + '/' + series+'_BubbleSegmented_NoEdge_Bin'+str(binvalue)+'_'+imifordir)
        elif sufix != None:
            image = imread(readdir + '/7_BubbleSegmented_NoEdge_'+sufix+'/' + series + '/' + series+'_BubbleSegmented_NoEdge_'+imifordir)
        else:
            image = imread(readdir + '/7_BubbleSegmented_NoEdge/' + series + '/' + series+'_BubbleSegmented_NoEdge_'+imifordir)
            
        # ShapeTensor data
        PropList = ShapeTensor(image, IncludeStrain=IncludeStrain)  
        
        # Save in TSV
        
        with open(path+'/'+series+'_Shape_'+imifordir+'.tsv', 'w', newline='') as csvfile:        
            writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            if IncludeStrain:
                writer.writerow(['Label',
                                 'z','y','x',
                                 'Vect1z','Vect1y','Vect1x','Vect2z','Vect2y','Vect2x','Vect3z','Vect3y','Vect3x', 
                                 'S1','S2','S3',
                                 'Szz','Syy','Sxx','Szy','Szx','Syx',
                                 'U1','U2','U3',
                                 'USzz','USyy','USxx','USzy','USzx','USyx',
                                 'VM'])
                for i in range(len(PropList[0])):
                    writer.writerow([PropList[0][i],
                                     PropList[1][i][0],PropList[1][i][1],PropList[1][i][2],

                                     PropList[2][i][0],PropList[2][i][1],PropList[2][i][2],
                                     PropList[3][i][0],PropList[3][i][1],PropList[3][i][2],
                                     PropList[4][i][0],PropList[4][i][1],PropList[4][i][2],
                                     PropList[5][i],PropList[6][i],PropList[7][i],
                                     PropList[8][i],PropList[9][i],PropList[10][i],PropList[11][i],PropList[12][i],PropList[13][i],

                                     PropList[14][i],PropList[15][i],PropList[16][i],
                                     PropList[17][i],PropList[18][i],PropList[19][i],PropList[20][i],PropList[21][i],PropList[22][i],
                                     PropList[23][i]])
            else:
                writer.writerow(['Label',
                                 'z','y','x',
                                 'Vect1z','Vect1y','Vect1x','Vect2z','Vect2y','Vect2x','Vect3z','Vect3y','Vect3x',
                                 'S1','S2','S3',
                                 'Szz','Syy','Sxx','Szy','Szx','Syx'])
                for i in range(len(PropList[0])):
                    writer.writerow([PropList[0][i],
                                     PropList[1][i][0],PropList[1][i][1],PropList[1][i][2],
                                     PropList[2][i][0],PropList[2][i][1],PropList[2][i][2],
                                     PropList[3][i][0],PropList[3][i][1],PropList[3][i][2],
                                     PropList[4][i][0],PropList[4][i][1],PropList[4][i][2],
                                     PropList[5][i],PropList[6][i],PropList[7][i],
                                     PropList[8][i],PropList[9][i],PropList[10][i],PropList[11][i],PropList[12][i],PropList[13][i]])
        if verbose:
            print('Image'+imifordir+' Nregions: ',len(PropList[0]), ':done')