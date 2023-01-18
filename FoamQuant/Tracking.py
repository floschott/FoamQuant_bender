def LabelTracking(Prop1,Prop2,searchbox=[-10,10,-10,10,-10,10],Volpercent=0.1):
    import numpy as np
    
    Lab1 = np.asarray(Prop1['lab'])
    Lab2 = np.asarray(Prop2['lab'])
    X1 = np.asarray(Prop1['x'])
    X2 = np.asarray(Prop2['x'])
    Y1 = np.asarray(Prop1['y'])
    Y2 = np.asarray(Prop2['y'])
    Z1 = np.asarray(Prop1['z'])
    Z2 = np.asarray(Prop2['z'])
    Vol1 = np.asarray(Prop1['vol'])
    Vol2 = np.asarray(Prop2['vol'])
    
    Tracklab=[]
    TrackX=[]
    TrackY=[]
    TrackZ=[]
    TrackVol=[]
    Countfound=[]
    Lostlab=[]
    LostX=[]
    LostY=[]
    LostZ=[]
    Countlost = 0
    
    Removedfromlab2 = []
    
    for i1 in range(len(Lab1)):
        lab1=Lab1[i1]
        x1=X1[i1]
        y1=Y1[i1]
        z1=Z1[i1]
        vol1=Vol1[i1]
        
        mindist = np.inf
        found = False
        Count = 0
        
        for i2 in range(len(Lab2)):
            lab2=Lab2[i2]
            x2=X2[i2]
            y2=Y2[i2]
            z2=Z2[i2]
            vol2=Vol2[i2]
            
            edge1 = [x1+searchbox[0], y1+searchbox[2], z1+searchbox[4]]
            edge2 = [x1+searchbox[1], y1+searchbox[3], z1+searchbox[5]]
            
            if x2>edge1[0] and x2<edge2[0] and y2>edge1[1] and y2<edge2[1] and z2>edge1[2] and z2<edge2[2]:
                
                if vol2 > (1-Volpercent)*vol1 and vol2 < (1+Volpercent)*vol1:
                    dist = np.sqrt(np.power(x2-x1,2)+np.power(y2-y1,2)+np.power(z2-z1,2))
                    
                    if dist < mindist:
                        mindist = dist
                        found = True
                        Slab2 = lab2
                        Sx2 = x2
                        Sy2 = y2
                        Sz2 = z2
                        Svol2 = vol2
                        Si2 = i2
                        Count+=1

        if found:
            Tracklab.append([lab1,Slab2])
            TrackX.append([x1,Sx2])
            TrackY.append([y1,Sy2])
            TrackZ.append([z1,Sz2])
            Countfound.append(Count)
            # remove matched label from Lab2 list
            Lab2 = np.delete(Lab2,Si2)
            X2 = np.delete(X2,Si2)
            Y2 = np.delete(Y2,Si2)
            Z2 = np.delete(Z2,Si2)
            Vol2 = np.delete(Vol2,Si2)
            Removedfromlab2.append(Slab2)

        else:
            Tracklab.append([lab1,-1])
            TrackX.append([x1,-1])
            TrackY.append([y1,-1])
            TrackZ.append([z1,-1])
            Countfound.append(Count)
            Countlost+=1
            Lostlab.append(lab1)
            LostX.append(x1)
            LostY.append(y1)
            LostZ.append(z1)
            
    print('Lost tracking:',Countlost,Countlost/len(Lab1)*100,'%')
    return Tracklab, TrackX, TrackY, TrackZ, Countfound, Lostlab, LostX, LostY, LostZ


def LabelTracking_Batch(nameread, namesave, dirread, dirsave, imrange, verbose=False, endread='.tsv', endsave='.tsv', n0=3,searchbox=[-10,10,-10,10,-10,10],Volpercent=0.1):
    
    import numpy as np
    import csv
    from FoamQuant.Tracking import LabelTracking
    from FoamQuant.Helper import strindex
    import os
    import pandas as pd
    
    #Check directory
    isExist = os.path.exists(dirsave)
    print('Path exist:', isExist)
    if not isExist:
        print('Saving path does not exist:\n', isExist)
        return
    
    LLostlab=[]; LLostX=[]; LLostY=[]; LLostZ=[]
    #Batch loop
    for i in range(len(imrange)-1):
        imi1=imrange[i]
        imi2=imrange[i]+1
        
        # image string index
        imifordir1 = strindex(imi1, n0)
        imifordir2 = strindex(imi2, n0)
        # read image
        Regprop1 = pd.read_csv(dirread+nameread+imifordir1+endread, sep='\t',engine="python",  quoting=csv.QUOTE_NONE)
        Regprop2 = pd.read_csv(dirread+nameread+imifordir2+endread, sep='\t',engine="python",  quoting=csv.QUOTE_NONE)
        
        Tracklab, TrackX, TrackY, TrackZ, Countfound,Lostlab, LostX, LostY, LostZ = LabelTracking(Regprop1,Regprop2,searchbox=searchbox,Volpercent=Volpercent)
        
        # Save as TSV
        with open(dirsave + namesave + imifordir1+'_'+imifordir2 + endsave, 'w', newline='') as csvfile:        
            writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['lab1','lab2','z1','z2','y1','y2','x1','x2','Count','dz','dy','dx'])
            for i in range(len(Tracklab)):
                writer.writerow([Tracklab[i][0],Tracklab[i][1],
                                 TrackZ[i][0],TrackZ[i][1],
                                 TrackY[i][0],TrackY[i][1],
                                 TrackX[i][0],TrackX[i][1],
                                 TrackZ[i][1]-TrackZ[i][0],TrackY[i][1]-TrackY[i][0],TrackX[i][1]-TrackX[i][0],
                                 Countfound[i]])
        if verbose:
            print(namesave+imifordir+': done')
            
        LLostlab.append(Lostlab); LLostX.append(LostX); LLostY.append(LostY); LLostZ.append(LostZ)
        
    return LLostlab, LLostX, LLostY, LLostZ
            
            
def Read_LabelTracking(nameread, dirread, imrange, verbose=False, endread='.tsv', n0=3):            
    import numpy as np 
    import pandas as pd
    from FoamQuant.Helper import strindex
    import csv
    
    lab1=[]; z1=[];y1=[];x1=[]
    lab2=[]; z2=[];y2=[];x2=[]
    Count=[]; dz=[];dy=[];dx=[]
    #Batch loop
    for i in range(len(imrange)-1):
        imi1=imrange[i]
        imi2=imrange[i]+1
        # image string index
        imifordir1 = strindex(imi1, n0)
        imifordir2 = strindex(imi2, n0)

        Regprops = pd.read_csv(dirread+nameread+ imifordir1+'_'+imifordir2 + endread, sep='\t',engine="python",  quoting=csv.QUOTE_NONE)
        lab1.append(np.asarray(Regprops['lab1']))
        z1.append(np.asarray(Regprops['z1']))
        y1.append(np.asarray(Regprops['y1']))
        x1.append(np.asarray(Regprops['x1']))
        lab2.append(np.asarray(Regprops['lab2']))
        z2.append(np.asarray(Regprops['z2']))
        y2.append(np.asarray(Regprops['y2']))
        x2.append(np.asarray(Regprops['x2']))
        Count.append(np.asarray(Regprops['Count']))
        dz.append(np.asarray(Regprops['dz']))
        dy.append(np.asarray(Regprops['dy']))
        dx.append(np.asarray(Regprops['dx']))
        
        if verbose:
                print(nameread+ imifordir1+'_'+imifordir2,': done')
        
    Properties={'lab1':lab1,'z1':z1,'y1':y1,'x1':x1,'lab2':lab2,'z2':z2,'y2':y2,'x2':x2, 'Count':Count, 'dz':dz, 'dy':dy, 'dx':dx}
    return Properties




def Combine_Tracking(nameread, dirread, imrange, verbose=False, endread='.tsv', n0=3):
    
    import numpy as np
    
    tracking = Read_LabelTracking(nameread, dirread, imrange, verbose=verbose, endread=endread, n0=n0)
    
    lab1 = tracking['lab1']
    lab2 = tracking['lab2']
    z1 = tracking['z1']
    z2 = tracking['z2']
    y1 = tracking['y1']
    y2 = tracking['y2']
    x1 = tracking['x1']
    x2 = tracking['x2']
    
    Nim = len(lab1)
    Nlab = len(lab1[0])
    
    Mlab = np.full((Nlab,Nim),np.nan)
    Mz = np.full((Nlab,Nim),np.nan)
    My = np.full((Nlab,Nim),np.nan)
    Mx = np.full((Nlab,Nim),np.nan)
    
    # 1st an 2nd im
    Mlab[:,0] = lab1[0]
    Mz[:,0] = z1[0]
    My[:,0] = y1[0]
    Mx[:,0] = x1[0]
    Mlab[:,1] = lab2[0]
    Mz[:,1] = z2[0]
    My[:,1] = y2[0]
    Mx[:,1] = x2[0]

    # next
    for imi in range(2,Nim):
        for labi in range(Nlab):
            for labnexti in range(len(lab1[imi-1])):

                if Mlab[labi,imi] != -1:
                    if Mlab[labi,imi-1] == lab1[imi-1][labnexti]:
                        Mlab[labi,imi] = lab2[imi-1][labnexti]
                        Mz[labi,imi] = z2[imi-1][labnexti]
                        My[labi,imi] = y2[imi-1][labnexti]
                        Mx[labi,imi] = x2[imi-1][labnexti]
                        
    combined = {'lab':Mlab,'z':Mz,'y':My,'x':Mx}
    
    return combined