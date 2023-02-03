def LabelTracking(Prop1,Prop2,searchbox=[-10,10,-10,10,-10,10],Volpercent=0.1):
    import numpy as np
    
    # Tracking properties
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
    
    # Other properties
    Rad1 = np.asarray(Prop1['rad'])
    Area1 = np.asarray(Prop1['area'])
    Sph1 = np.asarray(Prop1['sph'])
    Volfit1 = np.asarray(Prop1['volfit'])
    U1 = np.asarray(Prop1['U'])
    Type1 = np.asarray(Prop1['type'])
    Rad2 = np.asarray(Prop2['rad'])
    Area2 = np.asarray(Prop2['area'])
    Sph2 = np.asarray(Prop2['sph'])
    Volfit2 = np.asarray(Prop2['volfit'])
    U2 = np.asarray(Prop2['U'])
    Type2 = np.asarray(Prop2['type'])
    
    # Tracking lists
    Tracklab=[]
    TrackX=[]
    TrackY=[]
    TrackZ=[]
    TrackVol=[]
    # Lost lists
    Countfound=[]
    Lostlab=[]
    LostX=[]
    LostY=[]
    LostZ=[]
    Countlost = 0
    # Other properties lists
    Trackrad=[]
    Trackarea=[]
    Tracksph=[]
    Trackvolfit=[]
    TrackU=[]
    Tracktype=[]
    
    # Main loops
    Removedfromlab2 = []
    for i1 in range(len(Lab1)):
        lab1=Lab1[i1]
        x1=X1[i1]
        y1=Y1[i1]
        z1=Z1[i1]
        vol1=Vol1[i1]
        
        rad1 = Rad1[i1]
        area1 = Area1[i1]
        sph1 = Sph1[i1]
        volfit1 = Volfit1[i1]
        u1 = U1[i1]
        type1 = Type1[i1]
        
        mindist = np.inf
        found = False
        Count = 0
        
        for i2 in range(len(Lab2)):
            lab2=Lab2[i2]
            x2=X2[i2]
            y2=Y2[i2]
            z2=Z2[i2]
            vol2=Vol2[i2]
            rad2 = Rad2[i2]
            area2 = Area2[i2]
            sph2 = Sph2[i2]
            volfit2 = Volfit2[i2]
            u2 = U2[i2]
            type2 = Type2[i2]
            
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
                        
                        Svol2=vol2
                        Srad2 = rad2
                        Sarea2 = area2
                        Ssph2 = sph2
                        Svolfit2 = volfit2
                        Su2 = u2
                        Stype2 = type2

        if found:
            # Track properties
            Tracklab.append([lab1,Slab2])
            TrackX.append([x1,Sx2])
            TrackY.append([y1,Sy2])
            TrackZ.append([z1,Sz2])
            Countfound.append(Count)
            # Remove matched label from Lab2 list
            Lab2 = np.delete(Lab2,Si2)
            X2 = np.delete(X2,Si2)
            Y2 = np.delete(Y2,Si2)
            Z2 = np.delete(Z2,Si2)
            Vol2 = np.delete(Vol2,Si2)
            Removedfromlab2.append(Slab2)
            # Other tracked properties
            TrackVol.append([vol1, Svol2])
            Trackrad.append([rad1, Srad2])
            Trackarea.append([area1, Sarea2])
            Tracksph.append([sph1, Ssph2])
            Trackvolfit.append([volfit1, Svolfit2])
            TrackU.append([u1, Su2])
            Tracktype.append([type1, Stype2])

        else:
            # Track
            Tracklab.append([lab1,-1])
            TrackX.append([x1,-1])
            TrackY.append([y1,-1])
            TrackZ.append([z1,-1])
            Countfound.append(Count)
            # Lost
            Countlost+=1
            Lostlab.append(lab1)
            LostX.append(x1)
            LostY.append(y1)
            LostZ.append(z1)
            # Other tracked properties
            TrackVol.append([-1,-1])
            Trackrad.append([-1,-1])
            Trackarea.append([-1,-1])
            Tracksph.append([-1,-1])
            Trackvolfit.append([-1,-1])
            TrackU.append([-1,-1])
            Tracktype.append([-1,-1])
            
    print('Lost tracking:',Countlost,Countlost/len(Lab1)*100,'%')
    
    return Tracklab, TrackX, TrackY, TrackZ, Countfound, Lostlab, LostX, LostY, LostZ, TrackVol,Trackrad,Trackarea,Tracksph,Trackvolfit,TrackU,Tracktype


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
        
        Tracklab, TrackX, TrackY, TrackZ, Countfound, Lostlab, LostX, LostY, LostZ, TrackVol,Trackrad,Trackarea,Tracksph,Trackvolfit,TrackU,Tracktype = LabelTracking(Regprop1,Regprop2,searchbox=searchbox,Volpercent=Volpercent)
        
        # Save as TSV
        with open(dirsave + namesave + imifordir1+'_'+imifordir2 + endsave, 'w', newline='') as csvfile:        
            writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['lab1','lab2','z1','z2','y1','y2','x1','x2','dz','dy','dx','Count','vol1','vol2','rad1','rad2',
                             'area1','area2','sph1','sph2', 'volfit1','volfit2','U1','U2','type1','type2','Utype1','Utype2'])
            for i in range(len(Tracklab)):
                writer.writerow([Tracklab[i][0],Tracklab[i][1],
                                 TrackZ[i][0],TrackZ[i][1],
                                 TrackY[i][0],TrackY[i][1],
                                 TrackX[i][0],TrackX[i][1],
                                 TrackZ[i][1]-TrackZ[i][0],TrackY[i][1]-TrackY[i][0],TrackX[i][1]-TrackX[i][0],
                                 Countfound[i],
                                 TrackVol[i][0],
                                 TrackVol[i][1],
                                 Trackrad[i][0],
                                 Trackrad[i][1],
                                 Trackarea[i][0],
                                 Trackarea[i][1],
                                 Tracksph[i][0],
                                 Tracksph[i][1],
                                 Trackvolfit[i][0],
                                 Trackvolfit[i][1],
                                 TrackU[i][0],
                                 TrackU[i][1],
                                 Tracktype[i][0],
                                 Tracktype[i][1],
                                 TrackU[i][0]*Tracktype[i][0],
                                 TrackU[i][1]*Tracktype[i][1]])
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
    vol1=[]; rad1=[]; area1=[]; sph1=[]; volfit1=[]; U1=[]; types1=[]; Utype1=[]
    vol2=[]; rad2=[]; area2=[]; sph2=[]; volfit2=[]; U2=[]; types2=[]; Utype2=[]
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
        
        vol1.append(np.asarray(Regprops['vol1']))
        rad1.append(np.asarray(Regprops['rad1']))
        area1.append(np.asarray(Regprops['area1']))
        sph1.append(np.asarray(Regprops['sph1']))
        volfit1.append(np.asarray(Regprops['volfit1']))
        U1.append(np.asarray(Regprops['U1']))
        types1.append(np.asarray(Regprops['type1']))
        Utype1.append(np.asarray(Regprops['Utype1']))
        
        vol2.append(np.asarray(Regprops['vol2']))
        rad2.append(np.asarray(Regprops['rad2']))
        area2.append(np.asarray(Regprops['area2']))
        sph2.append(np.asarray(Regprops['sph2']))
        volfit2.append(np.asarray(Regprops['volfit2']))
        U2.append(np.asarray(Regprops['U2']))
        types2.append(np.asarray(Regprops['type2']))
        Utype2.append(np.asarray(Regprops['Utype2']))
        
        if verbose:
                print(nameread+ imifordir1+'_'+imifordir2,': done')
        
    Properties={'lab1':lab1,'z1':z1,'y1':y1,'x1':x1,'lab2':lab2,'z2':z2,'y2':y2,'x2':x2, 'Count':Count, 'dz':dz, 'dy':dy, 'dx':dx, 'Count':Count,'vol1':vol1,'rad1':rad1,'area1':area1,'sph1':sph1,'volfit1':volfit1,'U1':U1,'type1':types1,'Utype1':Utype1, 'vol2':vol2,'rad2':rad2,'area2':area2,'sph2':sph2,'volfit2':volfit2,'U2':U2,'type2':types2,'Utype2':Utype2}
    return Properties




def Combine_Tracking(nameread, dirread, imrange, verbose=False, endread='.tsv', n0=3):
    
    import numpy as np
    
    tracking = Read_LabelTracking(nameread, dirread, imrange, verbose=verbose, endread=endread, n0=n0)
    
    # Tracking properties
    lab1 = tracking['lab1']
    lab2 = tracking['lab2']
    z1 = tracking['z1']
    z2 = tracking['z2']
    y1 = tracking['y1']
    y2 = tracking['y2']
    x1 = tracking['x1']
    x2 = tracking['x2']
    
    #Other properties
    vz = tracking['dz']
    vy = tracking['dy']
    vx = tracking['dx']
    
    vol1 = tracking['vol1']
    rad1 = tracking['rad1']
    area1 = tracking['area1']
    sph1 = tracking['sph1']
    volfit1 = tracking['volfit1']
    U1 = tracking['U1']
    types1 = tracking['type1']
    Utype1 = tracking['Utype1']
    
    vol2 = tracking['vol2']
    rad2 = tracking['rad2']
    area2 = tracking['area2']
    sph2 = tracking['sph2']
    volfit2 = tracking['volfit2']
    U2 = tracking['U2']
    types2 = tracking['type2']
    Utype2 = tracking['Utype2']
    
    Nim = len(lab1)
    Nlab = len(lab1[0])
    
    # Tracking properties
    Mlab = np.full((Nlab,Nim),np.nan)
    Mz = np.full((Nlab,Nim),np.nan)
    My = np.full((Nlab,Nim),np.nan)
    Mx = np.full((Nlab,Nim),np.nan)

    #Other properties
    Mvz = np.full((Nlab,Nim),np.nan)
    Mvy = np.full((Nlab,Nim),np.nan)
    Mvx = np.full((Nlab,Nim),np.nan)
    Mvol = np.full((Nlab,Nim),np.nan)
    Mrad = np.full((Nlab,Nim),np.nan)
    Marea = np.full((Nlab,Nim),np.nan)
    Msph = np.full((Nlab,Nim),np.nan)
    Mvolfit = np.full((Nlab,Nim),np.nan)
    MU = np.full((Nlab,Nim),np.nan)
    Mtypes = np.full((Nlab,Nim),np.nan)
    MUtype = np.full((Nlab,Nim),np.nan)
    
    # 1st im
    Mlab[:,0] = lab1[0]
    Mz[:,0] = z1[0]
    My[:,0] = y1[0]
    Mx[:,0] = x1[0]
    
    Mvz[:,0] = 0
    Mvy[:,0] = 0
    Mvx[:,0] = 0
    Mvol[:,0] = vol1[0]
    Mrad[:,0] = rad1[0]
    Marea[:,0] = area1[0]
    Msph[:,0] = sph1[0]
    Mvolfit[:,0] = volfit1[0]
    MU[:,0] = U1[0]
    Mtypes[:,0] = types1[0]
    MUtype[:,0] = Utype1[0]
    
    # 2nd im
    Mlab[:,1] = lab2[0]
    Mz[:,1] = z2[0]
    My[:,1] = y2[0]
    Mx[:,1] = x2[0]
    
    Mvz[:,1] = vz[0]
    Mvy[:,1] = vy[0]
    Mvx[:,1] = vx[0]
    Mvol[:,1] = vol2[0]
    Mrad[:,1] = rad2[0]
    Marea[:,1] = area2[0]
    Msph[:,1] = sph2[0]
    Mvolfit[:,1] = volfit2[0]
    MU[:,1] = U2[0]
    Mtypes[:,1] = types2[0]
    MUtype[:,1] = Utype2[0]

    # next im
    for imi in range(2,Nim):
        for labi in range(Nlab):
            for labnexti in range(len(lab1[imi-1])):

                if Mlab[labi,imi] != -1:
                    if Mlab[labi,imi-1] == lab1[imi-1][labnexti]:
                        Mlab[labi,imi] = lab2[imi-1][labnexti]
                        Mz[labi,imi] = z2[imi-1][labnexti]
                        My[labi,imi] = y2[imi-1][labnexti]
                        Mx[labi,imi] = x2[imi-1][labnexti]
                        
                        Mvz[labi,imi] = vz[imi-1][labnexti]
                        Mvy[labi,imi] = vy[imi-1][labnexti]
                        Mvx[labi,imi] = vx[imi-1][labnexti]
                        Mvol[labi,imi] = vol2[imi-1][labnexti]
                        Mrad[labi,imi] = rad2[imi-1][labnexti]
                        Marea[labi,imi] = area2[imi-1][labnexti]
                        Msph[labi,imi] = sph2[imi-1][labnexti]
                        Mvolfit[labi,imi] = volfit2[imi-1][labnexti]
                        MU[labi,imi] = U2[imi-1][labnexti]
                        Mtypes[labi,imi] = types2[imi-1][labnexti]
                        MUtype[labi,imi] = Utype2[imi-1][labnexti]
                        
    combined = {'lab':Mlab,'z':Mz,'y':My,'x':Mx, 'dz':Mvz, 'dy':Mvy, 'dx':Mvx, 'vol':Mvol, 'rad':Mrad, 'area':Marea, 'sph':Msph, 'volfit':Mvolfit, 'U':MU, 'type':Mtypes, 'Utype':MUtype}
    
    return combined





def Save_Tracking(combined, namesave, dirsave, verbose=False, endsave='.csv'):
    
    Mz = combined['z']
    My = combined['y']
    Mx = combined['x']
    
    Mvz = combined['dz']
    Mvy = combined['dy']
    Mvx = combined['dx']
    Mvol = combined['vol']
    Mrad = combined['rad']
    Marea = combined['area']
    Msph = combined['sph']
    Mvolfit = combined['volfit']
    MU = combined['U']
    Mtypes = combined['type']
    MUtype = combined['Utype']
    
    import numpy as np
    import csv
    # Save as TSV
    with open(dirsave + namesave + endsave, 'w', newline='') as csvfile:        
        writer = csv.writer(csvfile, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['ti','labi','z','y','x', 'dz','dy','dx', 'vol','rad','area','sph','volfit','U','type','Utype'])
        
        for labi in range(len(Mz)):
            for ti in range(len(Mz[labi])):
                if Mz[labi][ti] != -1 and not np.isnan(Mz[labi][ti]):
                    z = Mz[labi][ti]
                    y = My[labi][ti]
                    x = Mx[labi][ti]
                    vz = Mvz[labi][ti]
                    vy = Mvy[labi][ti]
                    vx = Mvx[labi][ti]
                    vol = Mvol[labi][ti]
                    rad = Mrad[labi][ti]
                    area = Marea[labi][ti]
                    sph = Msph[labi][ti]
                    volfit = Mvolfit[labi][ti]
                    U = MU[labi][ti]
                    types = Mtypes[labi][ti]
                    Utype = MUtype[labi][ti]

                    writer.writerow([ti, labi, z,y,x, vz,vy,vx, vol,rad,area,sph,volfit,U,types,Utype])