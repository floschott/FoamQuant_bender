def Grid_Pavg(Coord, P, Range, N, NanFill=True, verbose=False, structured=True):
    """
    Return averaged scalar field over a 3D grid
    
    :param Coord: unstructured point coordinates (N,3)
    :type Coord: numpy array
    :param P: unstructured scalar array (N,1)
    :type P: numpy array
    :param Range: Averaging 3D ranges, such as [zmin,zmax,ymin,ymax,xmin,xmax]
    :type Range: numpy array
    :param N: Number of points along each dimension, such as [Nz,Ny,Nx]
    :type N: numpy array
    :param NanFill: If True fill with NaN when the Count is equal to 0 (else fill with zeros)
    :type NanFill: Bool
    :param verbose: If True, print the averaging Cartesian grid
    :type verbose: Bool
    :param structured: If True, return a structured scalar averaged field (Nz,Ny,Nx,1), else return an unstructured field (Nz*Ny*Nx,1)
    :type structured: Bool
    :return: [0] [grid0,grid1,grid2], [1] Grid coordinates, [2] Mean scalar field, [3] Std scalar field, [4] Count
    """
    
    import numpy as np
    
    Eps_1 = (Range[1]-Range[0])/(2*N[0])*1.0
    Eps_2 = (Range[3]-Range[2])/(2*N[1])*1.0
    Eps_3 = (Range[5]-Range[4])/(2*N[2])*1.0
    
    Grid1 = np.linspace(Range[0]+Eps_1, Range[1]-Eps_1, N[0])
    Grid2 = np.linspace(Range[2]+Eps_2, Range[3]-Eps_2, N[1])
    Grid3 = np.linspace(Range[4]+Eps_3, Range[5]-Eps_3, N[2])
    
    if verbose == True:
        print('Grid:\n', Grid1,'\n',Grid2,'\n',Grid3)
    
    Coord_avg = []
    Pavg = []
    Pstd = []
    Count = []
    
    if structured:
        if NanFill:
            Coord_avg = np.full((N[0],N[1],N[2],3), np.nan)
            Pavg = np.full((N[0],N[1],N[2]), np.nan)
            Pstd = np.full((N[0],N[1],N[2]), np.nan)
            Count = np.full((N[0],N[1],N[2]), np.nan)
        else:
            Coord_avg = np.zeros((N[0],N[1],N[2],3))
            Pavg = np.zeros((N[0],N[1],N[2]))
            Pstd = np.zeros((N[0],N[1],N[2]))
            Count = np.zeros((N[0],N[1],N[2]))
    
    for i1 in range(N[0]):
        for i2 in range(N[1]):
            for i3 in range(N[2]):
                if structured:
                    Coord_avg[i1,i2,i3,0] = Grid1[i1]
                    Coord_avg[i1,i2,i3,1] = Grid2[i2]
                    Coord_avg[i1,i2,i3,2] = Grid3[i3]
                else:
                    Coord_avg.append([Grid1[i1],Grid2[i2],Grid3[i3]])
                
                ListforAvg = []
                count = 0

                for i_obj in range(len(P)):
                    if Coord[i_obj][0] > Grid1[i1] - Eps_1 and Coord[i_obj][0] < Grid1[i1] + Eps_1:
                        if Coord[i_obj][1] > Grid2[i2] - Eps_2 and Coord[i_obj][1] < Grid2[i2] + Eps_2:
                            if Coord[i_obj][2] > Grid3[i3] - Eps_3 and Coord[i_obj][2] < Grid3[i3] + Eps_3:
                                ListforAvg.append(P[i_obj])
                                count+=1
                
                if structured:
                    if count>0:
                        Count[i1,i2,i3] = count
                        Pavg[i1,i2,i3] = np.nanmean(ListforAvg)                        
                        Pstd[i1,i2,i3] = np.nanstd(ListforAvg)
                        
                else:
                    Count.append(count)
                    if count>0:
                        Pavg.append(np.nanmean(ListforAvg))
                        Pstd.append(np.nanstd(ListforAvg))
                    else:
                        Pavg.append(np.nan)
                        Pstd.append(np.nan)
                    
    return [Grid1,Grid2,Grid3], np.asarray(Coord_avg), np.asarray(Pavg), np.asarray(Pstd), np.asarray(Count)



def Grid_Vavg(Coord, V, Range, N, NanFill=True, verbose=False, structured=True):
    """
    Return averaged vector field over a 3D grid
    
    :param Coord: unstructured point coordinates (N,3)
    :type Coord: numpy array
    :param P: unstructured vector array (N,3)
    :type P: numpy array
    :param Range: Averaging 3D ranges, such as [zmin,zmax,ymin,ymax,xmin,xmax]
    :type Range: numpy array
    :param N: Number of points along each dimension, such as [Nz,Ny,Nx]
    :type N: numpy array
    :param NanFill: If True fill with NaN when the Count is equal to 0 (else fill with zeros)
    :type NanFill: Bool
    :param verbose: If True, print the averaging Cartesian grid
    :type verbose: Bool
    :param structured: If True, return a structured vector averaged field (Nz,Ny,Nx,1), else return an unstructured field (Nz*Ny*Nx,3)
    :type structured: Bool
    :return: [0] [grid0,grid1,grid2], [1] Grid coordinates, [2] Mean vector field, [3] Std vector field, [4] Count
    """
    
    import numpy as np
    
    Eps_1 = (Range[1]-Range[0])/(2*N[0])*1.0
    Eps_2 = (Range[3]-Range[2])/(2*N[1])*1.0
    Eps_3 = (Range[5]-Range[4])/(2*N[2])*1.0
    
    Grid1 = np.linspace(Range[0]+Eps_1, Range[1]-Eps_1, N[0])
    Grid2 = np.linspace(Range[2]+Eps_2, Range[3]-Eps_2, N[1])
    Grid3 = np.linspace(Range[4]+Eps_3, Range[5]-Eps_3, N[2])
    
    if verbose == True:
        print('Grid:\n', Grid1,'\n',Grid2,'\n',Grid3)
    
    Coord_avg = []
    Vavg = []
    Vstd = []
    Count = []
    
    if structured:
        if NanFill:
            Coord_avg = np.full((N[0],N[1],N[2],3), np.nan)
            Vavg = np.full((N[0],N[1],N[2],3), np.nan)
            Vstd = np.full((N[0],N[1],N[2],3), np.nan)
            Count = np.full((N[0],N[1],N[2]), np.nan)
        else:
            Coord_avg = np.zeros((N[0],N[1],N[2],3))
            Vavg = np.zeros((N[0],N[1],N[2],3))
            Vstd = np.zeros((N[0],N[1],N[2],3))
            Count = np.zeros((N[0],N[1],N[2]))
    
    for i1 in range(N[0]):
        for i2 in range(N[1]):
            for i3 in range(N[2]):
                if structured:
                    Coord_avg[i1,i2,i3,0] = Grid1[i1]
                    Coord_avg[i1,i2,i3,1] = Grid2[i2]
                    Coord_avg[i1,i2,i3,2] = Grid3[i3]
                else:
                    Coord_avg.append([Grid1[i1],Grid2[i2],Grid3[i3]])
                
                ListforAvg_1 = []
                ListforAvg_2 = []
                ListforAvg_3 = []
                count = 0

                for i_obj in range(len(V)):
                    if Coord[i_obj][0] > Grid1[i1] - Eps_1 and Coord[i_obj][0] < Grid1[i1] + Eps_1:
                        if Coord[i_obj][1] > Grid2[i2] - Eps_2 and Coord[i_obj][1] < Grid2[i2] + Eps_2:
                            if Coord[i_obj][2] > Grid3[i3] - Eps_3 and Coord[i_obj][2] < Grid3[i3] + Eps_3:
                                ListforAvg_1.append(V[i_obj][0])
                                ListforAvg_2.append(V[i_obj][1])
                                ListforAvg_3.append(V[i_obj][2])
                                count+=1
                
                if structured:
                    if count>0:
                        Count[i1,i2,i3] = count
                        Vavg[i1,i2,i3,0] = np.nanmean(ListforAvg_1)
                        Vavg[i1,i2,i3,1] = np.nanmean(ListforAvg_2)
                        Vavg[i1,i2,i3,2] = np.nanmean(ListforAvg_3)
                        Vstd[i1,i2,i3,0] = np.nanstd(ListforAvg_1)
                        Vstd[i1,i2,i3,1] = np.nanstd(ListforAvg_2)
                        Vstd[i1,i2,i3,2] = np.nanstd(ListforAvg_3)
                else:
                    Count.append(count)
                    if count>0:
                        Vavg.append([np.nanmean(ListforAvg_1),np.nanmean(ListforAvg_2),np.nanmean(ListforAvg_3)])
                        Vstd.append([np.nanstd(ListforAvg_1),np.nanstd(ListforAvg_2),np.nanstd(ListforAvg_3)])
                    else:
                        Vavg.append([np.nan,np.nan,np.nan])
                        Vstd.append([np.nan,np.nan,np.nan])
                    
    return [Grid1,Grid2,Grid3], np.asarray(Coord_avg), np.asarray(Vavg), np.asarray(Vstd), np.asarray(Count)



def Grid_Tavg(Coord, T, Range, N, NanFill=True, verbose=False, structured=True):
    """
    Return averaged tensor field over a 3D grid
    
    :param Coord: unstructured point coordinates (N,3)
    :type Coord: numpy array
    :param P: unstructured tensor array (N,3,3)
    :type P: numpy array
    :param Range: Averaging 3D ranges, such as [zmin,zmax,ymin,ymax,xmin,xmax]
    :type Range: numpy array
    :param N: Number of points along each dimension, such as [Nz,Ny,Nx]
    :type N: numpy array
    :param NanFill: If True fill with NaN when the Count is equal to 0 (else fill with zeros)
    :type NanFill: Bool
    :param verbose: If True, print the averaging Cartesian grid
    :type verbose: Bool
    :param structured: If True, return a structured tensor averaged field (Nz,Ny,Nx,3,3), else return an unstructured field (Nz*Ny*Nx,3,3)
    :type structured: Bool
    :return: [0] [grid0,grid1,grid2], [1] Grid coordinates, [2] Mean tensor field, [3] Std tensor field, [4] Count
    """
    
    import numpy as np
    
    Eps_1 = (Range[1]-Range[0])/(2*N[0])*1.0
    Eps_2 = (Range[3]-Range[2])/(2*N[1])*1.0
    Eps_3 = (Range[5]-Range[4])/(2*N[2])*1.0
    
    Grid1 = np.linspace(Range[0]+Eps_1, Range[1]-Eps_1, N[0])
    Grid2 = np.linspace(Range[2]+Eps_2, Range[3]-Eps_2, N[1])
    Grid3 = np.linspace(Range[4]+Eps_3, Range[5]-Eps_3, N[2])
    
    if verbose == True:
        print('Grid:\n', Grid1,'\n',Grid2,'\n',Grid3)
    
    Coord_avg = []
    Tavg = []
    Tstd = []
    Count = []
    
    if structured:
        if NanFill:
            Coord_avg = np.full((N[0],N[1],N[2],3,3), np.nan)
            Tavg = np.full((N[0],N[1],N[2],3,3), np.nan)
            Tstd = np.full((N[0],N[1],N[2],3,3), np.nan)
            Count = np.full((N[0],N[1],N[2]), np.nan)
        else:
            Coord_avg = np.zeros((N[0],N[1],N[2],3,3))
            Tavg = np.zeros((N[0],N[1],N[2],3,3))
            Tstd = np.zeros((N[0],N[1],N[2],3,3))
            Count = np.zeros((N[0],N[1],N[2]))
    
    for i1 in range(N[0]):
        for i2 in range(N[1]):
            for i3 in range(N[2]):
                if structured:
                    Coord_avg[i1,i2,i3,0] = Grid1[i1]
                    Coord_avg[i1,i2,i3,1] = Grid2[i2]
                    Coord_avg[i1,i2,i3,2] = Grid3[i3]
                else:
                    Coord_avg.append([Grid1[i1],Grid2[i2],Grid3[i3]])
                
                ListforAvg_11 = []
                ListforAvg_22 = []
                ListforAvg_33 = []
                ListforAvg_12 = []
                ListforAvg_13 = []
                ListforAvg_23 = []
                ListforAvg_21 = []
                ListforAvg_31 = []
                ListforAvg_32 = []
                count = 0

                for i_obj in range(len(T)):
                    if Coord[i_obj][0] > Grid1[i1] - Eps_1 and Coord[i_obj][0] < Grid1[i1] + Eps_1:
                        if Coord[i_obj][1] > Grid2[i2] - Eps_2 and Coord[i_obj][1] < Grid2[i2] + Eps_2:
                            if Coord[i_obj][2] > Grid3[i3] - Eps_3 and Coord[i_obj][2] < Grid3[i3] + Eps_3:
                                ListforAvg_11.append(T[i_obj][0][0])
                                ListforAvg_22.append(T[i_obj][1][1])
                                ListforAvg_33.append(T[i_obj][2][2])
                                ListforAvg_12.append(T[i_obj][0][1])
                                ListforAvg_13.append(T[i_obj][0][2])
                                ListforAvg_23.append(T[i_obj][1][2])
                                ListforAvg_21.append(T[i_obj][1][0])
                                ListforAvg_31.append(T[i_obj][2][0])
                                ListforAvg_32.append(T[i_obj][2][1])
                                count+=1
                
                if structured:
                    if count>0:
                        Count[i1,i2,i3] = count
                        Tavg[i1,i2,i3,0,0] = np.nanmean(ListforAvg_11)
                        Tavg[i1,i2,i3,1,1] = np.nanmean(ListforAvg_22)
                        Tavg[i1,i2,i3,2,2] = np.nanmean(ListforAvg_33)
                        Tavg[i1,i2,i3,0,1] = np.nanmean(ListforAvg_12)
                        Tavg[i1,i2,i3,0,2] = np.nanmean(ListforAvg_13)
                        Tavg[i1,i2,i3,1,2] = np.nanmean(ListforAvg_23)
                        Tavg[i1,i2,i3,1,0] = np.nanmean(ListforAvg_21)
                        Tavg[i1,i2,i3,2,0] = np.nanmean(ListforAvg_31)
                        Tavg[i1,i2,i3,2,1] = np.nanmean(ListforAvg_32)
                        
                        Tstd[i1,i2,i3,0,0] = np.nanstd(ListforAvg_11)
                        Tstd[i1,i2,i3,1,1] = np.nanstd(ListforAvg_22)
                        Tstd[i1,i2,i3,2,2] = np.nanstd(ListforAvg_33)
                        Tstd[i1,i2,i3,0,1] = np.nanstd(ListforAvg_12)
                        Tstd[i1,i2,i3,0,2] = np.nanstd(ListforAvg_13)
                        Tstd[i1,i2,i3,1,2] = np.nanstd(ListforAvg_23)
                        Tstd[i1,i2,i3,1,0] = np.nanstd(ListforAvg_21)
                        Tstd[i1,i2,i3,2,0] = np.nanstd(ListforAvg_31)
                        Tstd[i1,i2,i3,2,1] = np.nanstd(ListforAvg_32)
                else:
                    Count.append(count)
                    if count>0:
                        Tavg.append([[np.nanmean(ListforAvg_11),np.nanmean(ListforAvg_12),np.nanmean(ListforAvg_13)],
                                     [np.nanmean(ListforAvg_21),np.nanmean(ListforAvg_22),np.nanmean(ListforAvg_23)],
                                     [np.nanmean(ListforAvg_31),np.nanmean(ListforAvg_32),np.nanmean(ListforAvg_33)]])
                        Tstd.append([[np.nanstd(ListforAvg_11),np.nanstd(ListforAvg_12),np.nanstd(ListforAvg_13)],
                                     [np.nanstd(ListforAvg_21),np.nanstd(ListforAvg_22),np.nanstd(ListforAvg_23)],
                                     [np.nanstd(ListforAvg_31),np.nanstd(ListforAvg_32),np.nanstd(ListforAvg_33)]])
                    else:
                        Tavg.append([[np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan]])
                        Tstd.append([[np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan]])
                    
    return [Grid1,Grid2,Grid3], np.asarray(Coord_avg), np.asarray(Tavg), np.asarray(Tstd), np.asarray(Count)