def Azimuth(X,Y):
    """
    Return the aziuthal angle of a point from its coordinates x and y
    
    :param X: coordinate x
    :type X: float
    :param Y: coordinate y
    :type Y: float
    :return: float, cylindrical azimuthal angle phi
    """
    
    import numpy as np
    if X > 0:
        phi_Cyl = np.arctan(Y/X)
    elif X < 0 and Y >= 0:
        phi_Cyl = np.arctan(Y/X)+np.pi
    elif X < 0 and Y <= 0:
        phi_Cyl = np.arctan(Y/X)-np.pi
    elif X == 0 and Y > 0:
        phi_Cyl = np.pi/2
    elif X == 0 and Y < 0:
        phi_Cyl = -np.pi/2
    elif X == 0 and Y == 0:
        phi_Cyl = 0
    return phi_Cyl

def CylR(X,Y):
    """
    Return the cylindrical radius r of a point from its coordinates x and y
    
    :param X: coordinate x
    :type X: float
    :param Y: coordinate y
    :type Y: float
    :return: float, cylindrical radius r
    """
    
    import numpy as np
    return np.sqrt(np.power(X,2)+np.power(Y,2))

def Pzyx2zphir(Azi):
    """
    Return the cylindrical passage matrix from the cylindrical aziuthal angle Azi
    
    :param phi: azimuthal angle Azi
    :type phi: float
    :return: Passage matrix from (z,y,x) to (z,phi,r), (3,3) numpy array
    """
    
    import numpy as np
    return np.asarray([[1, 0, 0], 
                       [0, np.cos(Azi), -np.sin(Azi)], 
                       [0, np.sin(Azi), np.cos(Azi)]])

def CylCoord(Coord_Cartesian, CoordAxis = [0,1008,1008], CylAxisZ = [1,0,0], CylAxisY = [0,1,0], CylAxisX = [0,0,1]):
    """
    Return the cylindrical coordinate of a point from its cartesian coordinates
    
    :param Coord_Cartesian: cartesian coordinates in the image basis [z,y,x]
    :type Coord_Cartesian: numpy array
    :param CoordAxis: Coordinates of the axis of rotation in the image basis [z,y,x]
    :type CoordAxis: numpy array
    :param CylAxisZ: Direction vector of the axis of rotation Z expressed in the image basis [z,y,x]
    :type CylAxisZ: numpy array
    :param CylAxisY: Direction vector of the axis Y expressed in the image basis [z,y,x]
    :type CylAxisY: numpy array
    :param CylAxisX: Direction vector of the axis X expressed in the image basis [z,y,x]
    :type CylAxisX: numpy array
    :return: cylindrical coordinates [r,phi,z]
    """
    
    import numpy as np
    # Cartesian image -> Cartesian Cyl axis
    
    Z, Y, X =  np.asarray([CylAxisZ,
                           CylAxisY,
                           CylAxisX]) @ [Coord_Cartesian[0]-CoordAxis[0],
                                         Coord_Cartesian[1]-CoordAxis[1],
                                         Coord_Cartesian[2]-CoordAxis[2]]
    # Cartesian Cyl axis -> Cylindrical
    r_Cyl = CylR(X,Y)
    phi_Cyl = Azimuth(X,Y)
    return [r_Cyl, phi_Cyl, Z]

def Cartesian2Cylindrical_Point(Coord_Cartesian, CoordAxis = [0,1008,1008], CylAxisZ = [1,0,0],CylAxisY = [0,1,0],CylAxisX = [0,0,1]):
    """
    Return unstructured cylindrical coordinates array (N,3) from unstructured cartesian coordinates (N,3)
    
    :param Coord_Cartesian: unstructured cartesian coordinates, expressed in the image basis [z,y,x]
    :type Coord_Cartesian: numpy array
    :param CoordAxis: Coordinates of the axis of rotation in the image basis [z,y,x]
    :type CoordAxis: numpy array
    :param CylAxisZ: Direction vector of the axis of rotation Z expressed in the image basis [z,y,x]
    :type CylAxisZ: numpy array
    :param CylAxisY: Direction vector of the axis Y expressed in the image basis [z,y,x]
    :type CylAxisY: numpy array
    :param CylAxisX: Direction vector of the axis X expressed in the image basis [z,y,x]
    :type CylAxisX: numpy array
    :return: unstructured cylindrical coordinates (N,3), expressed in the cylindrical basis [r,phi,z]
    """
    
    import numpy as np
    Coord_Cyl=[]
    for Coord in Coord_Cartesian:
        r,phi,z = CylCoord(Coord, CoordAxis, CylAxisZ, CylAxisY, CylAxisX)
        Coord_Cyl.append([r,phi,z])
    return np.asarray(Coord_Cyl)

def Cartesian2Cylindrical_Vector(Coord_Cartesian, V_Cartesian, CoordAxis = [0,1008,1008], CylAxisZ = [1,0,0],CylAxisY = [0,1,0],CylAxisX = [0,0,1]):
    """
    Return unstructured cylindrical coordinates (N,3) and vectors (N,3) from unstructured cartesian coordinates (N,3) and vectors (N,3)
    
    :param Coord_Cartesian: unstructured cartesian coordinates, expressed in the image basis [z,y,x]
    :type Coord_Cartesian: numpy array
    :param CoordAxis: Coordinates of the axis of rotation in the image basis [z,y,x]
    :type CoordAxis: numpy array
    :param CylAxisZ: Direction vector of the axis of rotation Z expressed in the image basis [z,y,x]
    :type CylAxisZ: numpy array
    :param CylAxisY: Direction vector of the axis Y expressed in the image basis [z,y,x]
    :type CylAxisY: numpy array
    :param CylAxisX: Direction vector of the axis X expressed in the image basis [z,y,x]
    :type CylAxisX: numpy array
    :return: unstructured cylindrical coordinates [0] and vectors [1], expressed in the cylindrical basis [r,phi,z]
    """
    
    import numpy as np
    Coord_Cyl=[]; V_Cyl=[]
    for i in range(len(Coord_Cartesian)):
        ## COORD
        r_Cyl,phi_Cyl,z_Cyl = CylCoord(Coord_Cartesian[i], CoordAxis, CylAxisZ, CylAxisY, CylAxisX)
        Coord_Cyl.append([r_Cyl,phi_Cyl,z_Cyl])
        ## VECTOR
        # Cartesian image -> Cartesian Cyl axis
        VZ, VY, VX = np.asarray([CylAxisZ,
                                 CylAxisY,
                                 CylAxisX]) @ [V_Cartesian[i][0], 
                                               V_Cartesian[i][1], 
                                               V_Cartesian[i][2]]
        # Cartesian Cyl axis -> Cylindrical
        PZYXzphir = Pzyx2zphir(phi_Cyl)
        Vz_Cyl, Vphi_Cyl, Vr_Cyl = PZYXzphir @ [VZ, VY, VX]
        V_Cyl.append([Vr_Cyl, Vphi_Cyl, Vz_Cyl])
    return np.asarray(Coord_Cyl), np.asarray(V_Cyl)


def Cartesian2Cylindrical_Tensor(Coord_Cartesian, T_Cartesian, CoordAxis = [0,1008,1008], CylAxisZ = [1,0,0],CylAxisY = [0,1,0],CylAxisX = [0,0,1]):
    """
    Return unstructured cylindrical coordinates (N,3) and tensors (N,3,3) from unstructured cartesian coordinates (N,3) and tensors (N,3,3)
    
    :param Coord_Cartesian: unstructured cartesian coordinates, expressed in the image basis [z,y,x]
    :type Coord_Cartesian: numpy array
    :param CoordAxis: Coordinates of the axis of rotation in the image basis [z,y,x]
    :type CoordAxis: numpy array
    :param CylAxisZ: Direction vector of the axis of rotation Z expressed in the image basis [z,y,x]
    :type CylAxisZ: numpy array
    :param CylAxisY: Direction vector of the axis Y expressed in the image basis [z,y,x]
    :type CylAxisY: numpy array
    :param CylAxisX: Direction vector of the axis X expressed in the image basis [z,y,x]
    :type CylAxisX: numpy array
    :return: unstructured cylindrical coordinates [0] and tensors [1], expressed in the cylindrical basis [r,phi,z]
    """
    
    import numpy as np
    Coord_Cyl=[]; T_Cyl=[]
    for i in range(len(Coord_Cartesian)):
        ## COORD
        r_Cyl,phi_Cyl,z_Cyl = CylCoord(Coord_Cartesian[i], CoordAxis, CylAxisZ, CylAxisY, CylAxisX)
        Coord_Cyl.append([r_Cyl,phi_Cyl,z_Cyl])
        ## TENSOR
        # Cartesian image -> Cartesian Cyl axis        
        T_Cartesian_ZYX = np.transpose(PzyxZYX) @ T_Cartesian[i] @ PzyxZYX
        # Cartesian Cyl axis -> Cylindrical
        PZYXzphir = Pzyx2zphir(phi_Cyl)        
        T_Cyl_zphir = np.transpose(PZYXzphir) @ T_Cartesian_ZYX @ PZYXzphir 
        T_Cyl.append(T_Cyl_zphir) 
    return np.asarray(Coord_Cyl), np.asarray(T_Cyl)



########################################################################################
def SpheR(X,Y,Z):
    """
    Return the spherical radius of a point from its coordinates x, y and z
    
    :param X: coordinate x
    :type X: float
    :param Y: coordinate y
    :type Y: float
    :param Z: coordinate z
    :type Z: float
    :return: float, spherical radius r
    """
    
    import numpy as np
    return np.sqrt(np.power(X,2)+np.power(Y,2)+np.power(Z,2))


def Polar(Z,spheR):
    """
    Return the spherical polar angle of a point from its coordinates x, y and z
    
    :param X: coordinate x
    :type X: float
    :param Y: coordinate y
    :type Y: float
    :param Z: coordinate z
    :type Z: float
    :return: float, spherical radius r
    """
    
    import numpy as np
    return np.arccos(Z)/spheR

def Pzyx2phithetar(Azi, Polar):
    """
    Return the spherical passage matrix, from the spherical aziuthal and polar angles Azi and Polar
    
    :param Azi: azimuthal angle Azi
    :type Azi: float
    :param Polar: polar angle Polar
    :type Polar: float
    :return: Passage matrix from (z,y,x) to (phi,theta,r), (3,3) numpy array
    """
    
    import numpy as np
    return np.asarray([[0,              np.cos(Azi),              -np.sin(Azi)],
                       [-np.sin(Polar), np.cos(Polar)*np.sin(Azi), np.cos(Polar)*np.cos(Azi)], 
                       [ np.cos(Polar), np.sin(Polar)*np.sin(Azi), np.sin(Polar)*np.cos(Azi)]])

def SpheCoord(Coord_Cartesian, CoordAxis = [0,1008,1008], CylAxisZ = [1,0,0], CylAxisY = [0,1,0], CylAxisX = [0,0,1]):
    """
    Return the spherical coordinates of a point from its cartesian coordinates
    
    :param Coord_Cartesian: cartesian coordinates in the image basis [z,y,x]
    :type Coord_Cartesian: numpy array
    :param CoordAxis: Coordinates of the axis of rotation in the image basis [z,y,x]
    :type CoordAxis: numpy array
    :param CylAxisZ: Direction vector of the axis of rotation Z expressed in the image basis [z,y,x]
    :type CylAxisZ: numpy array
    :param CylAxisY: Direction vector of the axis Y expressed in the image basis [z,y,x]
    :type CylAxisY: numpy array
    :param CylAxisX: Direction vector of the axis X expressed in the image basis [z,y,x]
    :type CylAxisX: numpy array
    :return: spherical coordinates [r,phi,z]
    """
    
    import numpy as np
    # Cartesian image -> Cartesian Cyl axis
    Z, Y, X =  np.asarray([CylAxisZ,
                           CylAxisY,
                           CylAxisX]) @ [Coord_Cartesian[0]-CoordAxis[0],
                                         Coord_Cartesian[1]-CoordAxis[1],
                                         Coord_Cartesian[2]-CoordAxis[2]]
    # Cartesian Cyl axis -> Cylindrical
    r_Sph = SpheR(X,Y,Z)
    Polar_Sph = Polar(Z,r_Sph)
    Azi_Sph = Azimuth(X,Y)
    return [r_Sph, Polar_Sph, Azi_Sph]

def Cartesian2Spherical_Point(Coord_Cartesian, CoordAxis = [0,1008,1008], CylAxisZ = [1,0,0],CylAxisY = [0,1,0],CylAxisX = [0,0,1]):
    """
    Return unstructured spherical coordinates array (N,3) from unstructured cartesian coordinates (N,3)
    
    :param Coord_Cartesian: unstructured cartesian coordinates, expressed in the image basis [z,y,x]
    :type Coord_Cartesian: numpy array
    :param CoordAxis: Coordinates of the axis of rotation in the image basis [z,y,x]
    :type CoordAxis: numpy array
    :param CylAxisZ: Direction vector of the axis of rotation Z expressed in the image basis [z,y,x]
    :type CylAxisZ: numpy array
    :param CylAxisY: Direction vector of the axis Y expressed in the image basis [z,y,x]
    :type CylAxisY: numpy array
    :param CylAxisX: Direction vector of the axis X expressed in the image basis [z,y,x]
    :type CylAxisX: numpy array
    :return: unstructured cylindrical coordinates (N,3), expressed in the cylindrical basis [r,phi,z]
    """
    
    import numpy as np
    Coord_Cyl=[]
    for Coord in Coord_Cartesian:
        r,theta,phi = SpheCoord(Coord, CoordAxis, CylAxisZ, CylAxisY, CylAxisX)
        Coord_Cyl.append([r,theta,phi])
    return np.asarray(Coord_Cyl)

def Cartesian2Spherical_Vector(Coord_Cartesian, V_Cartesian, CoordAxis = [0,1008,1008], CylAxisZ = [1,0,0],CylAxisY = [0,1,0],CylAxisX = [0,0,1]):
    """
    Return unstructured spherical coordinates (N,3) and vectors (N,3) from unstructured cartesian coordinates (N,3) and vectors (N,3)
    
    :param Coord_Cartesian: unstructured cartesian coordinates, expressed in the image basis [z,y,x]
    :type Coord_Cartesian: numpy array
    :param CoordAxis: Coordinates of the axis of rotation in the image basis [z,y,x]
    :type CoordAxis: numpy array
    :param CylAxisZ: Direction vector of the axis of rotation Z expressed in the image basis [z,y,x]
    :type CylAxisZ: numpy array
    :param CylAxisY: Direction vector of the axis Y expressed in the image basis [z,y,x]
    :type CylAxisY: numpy array
    :param CylAxisX: Direction vector of the axis X expressed in the image basis [z,y,x]
    :type CylAxisX: numpy array
    :return: unstructured cylindrical coordinates [0] and vectors [1], expressed in the cylindrical basis [r,phi,z]
    """
    
    import numpy as np
    Coord_Sphe=[]; V_Sphe=[]
    for i in range(len(Coord_Cartesian)):
        ## COORD
        r,theta,phi = SpheCoord(Coord_Cartesian[i], CoordAxis, CylAxisZ, CylAxisY, CylAxisX)
        Coord_Cyl.append([r,theta,phi])
        ## VECTOR
        # Cartesian image -> Cartesian Cyl axis
        VZ, VY, VX = np.asarray([CylAxisZ,
                                 CylAxisY,
                                 CylAxisX]) @ [V_Cartesian[i][0], 
                                               V_Cartesian[i][1], 
                                               V_Cartesian[i][2]]
        # Cartesian Cyl axis -> Cylindrical
        PZYXphithetar = Pzyx2phithetar(phi,theta)
        Vphi, Vtheta, Vr = PZYXphithetar @ [VZ, VY, VX]
        V_Sphe.append([Vr, Vtheta, Vphi])
    return np.asarray(Coord_Sphe), np.asarray(V_Sphe)


def Cartesian2Spherical_Tensor(Coord_Cartesian, T_Cartesian, CoordAxis = [0,1008,1008], CylAxisZ = [1,0,0],CylAxisY = [0,1,0],CylAxisX = [0,0,1]):
    """
    Return unstructured spherical coordinates (N,3) and tensors (N,3,3) from unstructured cartesian coordinates (N,3) and tensors (N,3,3)
    
    :param Coord_Cartesian: unstructured cartesian coordinates, expressed in the image basis [z,y,x]
    :type Coord_Cartesian: numpy array
    :param CoordAxis: Coordinates of the axis of rotation in the image basis [z,y,x]
    :type CoordAxis: numpy array
    :param CylAxisZ: Direction vector of the axis of rotation Z expressed in the image basis [z,y,x]
    :type CylAxisZ: numpy array
    :param CylAxisY: Direction vector of the axis Y expressed in the image basis [z,y,x]
    :type CylAxisY: numpy array
    :param CylAxisX: Direction vector of the axis X expressed in the image basis [z,y,x]
    :type CylAxisX: numpy array
    :return: unstructured cylindrical coordinates [0] and tensors [1], expressed in the cylindrical basis [r,phi,z]
    """
    
    import numpy as np
    Coord_Sphe=[]; T_Sphe=[]
    for i in range(len(Coord_Cartesian)):
        ## COORD
        r,theta,phi = SpheCoord(Coord_Cartesian[i], CoordAxis, CylAxisZ, CylAxisY, CylAxisX)
        Coord_Cyl.append([r,theta,phi])
        ## TENSOR
        # Cartesian image -> Cartesian Cyl axis        
        T_ZYX = np.transpose(PzyxZYX) @ T_Cartesian[i] @ PzyxZYX
        # Cartesian Cyl axis -> Cylindrical
        PZYXphithetar = Pzyx2phithetar(phi,theta)        
        T_zphir = np.transpose(PZYXphithetar) @ T_ZYX @ PZYXphithetar 
        T_Sphe.append(T_zphir) 
    return np.asarray(Coord_Sphe), np.asarray(T_Sphe)