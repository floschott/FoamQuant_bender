def writeGlyphsVTK(coordinates,
                   pointData,
                   fileName='SavedGlyphs.vtk'):
    """
    Slightly modified writeGlyphsVTK function from the SPAM package.
    
    Write a plain text glyphs vtk.
    
    :param coordinates: Coordinates of the centre of all ``n`` glyphs 
    :type coordinates: (N,3) array of float
    :param pointData: (N,1) arrays for scalar values, (N,3) for vector values and (N,3) for tensor values
    :type pointData: dict ``{'field1name':field1,'field2name':field2, ...}``
    :param fileName: Optional name of the output file. By default='.vtk'
    :type fileName: int
    :return: str index
    """
    
    # check dimensions
    import six
    import numpy
    dimension = coordinates.shape[0]
    
    if len(pointData):
        for k, v in six.iteritems(pointData):
            if dimension != v.shape[0]:
                print('spam.helpers.writeGlyphsVTK() Inconsistent point field sizes {} != {}'.format(dimension, v.shape[0]))
                return 0
    else:
        print('spam.helpers.writeGlyphsVTK() Empty files. Not writing {}'.format(fileName))
        return

    with open(fileName, 'w') as f:
        # header
        f.write('# vtk DataFile Version 2.0\n')
        # f.write('VTK file from spam: {}\n'.format(fileName))
        f.write('Unstructured grid legacy vtk file with point scalar data\n')
        f.write('ASCII\n\n')

        # coordinates
        f.write('DATASET UNSTRUCTURED_GRID\n')
        f.write('POINTS {:.0f} float\n'.format(dimension))
        for coord in coordinates:
            f.write('    {} {} {}\n'.format(*reversed(coord)))

        f.write('\n')

        # pointData
        if len(pointData) == 1:
            f.write('POINT_DATA {}\n\n'.format(dimension))
            _writeFieldInVtk(pointData, f, flat=True)
        elif len(pointData) > 1:
            f.write('POINT_DATA {}\n\n'.format(dimension))
            for k in pointData:
                _writeFieldInVtk({k: pointData[k]}, f, flat=True)
        f.write('\n')
        
        
def _writeFieldInVtk(data, f, flat=False):
    """
    Slightly modified writeGlyphsVTK function from the SPAM package.
    Private helper function for writing vtk fields
    """

    for key in data:
        field = data[key]
        #print(len(field.shape))
        #print(field.shape)

        if flat:
            # SCALAR flatten (n by 1)
            if(len(field.shape) == 1):
                print('flatten scalar')
                f.write('SCALARS {} float\n'.format(key.replace(" ", "_")))
                f.write('LOOKUP_TABLE default\n')
                for item in field:
                    f.write('    {}\n'.format(item))
                f.write('\n')

            # VECTORS flatten (n by 3)
            elif(len(field.shape) == 2 and field.shape[1] == 3):
                print('flatten vector')
                f.write('VECTORS {} float\n'.format(key.replace(" ", "_")))
                for item in field:
                    f.write('    {} {} {}\n'.format(*reversed(item)))
                f.write('\n')
                
                
            # TENSORS flatten (n by 3 by 3)
            elif(len(field.shape) == 3 and field.shape[1] * field.shape[2] == 9):
                print('flatten tensor')
                f.write('TENSORS {} float\n'.format(key.replace(" ", "_")))
                for item in field:
                    f.write('    {} {} {}\n    {} {} {}\n    {} {} {}\n\n'.format(item[0,0],item[0,1],item[0,2],
                                                                                  item[1,0],item[1,1],item[1,2],
                                                                                  item[2,0],item[2,1],item[2,2]))
                f.write('\n')  

        else:
            # SCALAR not flatten (n1 by n2 by n3)
            if(len(field.shape) == 3):
                print('not flatten scalar')
                f.write('SCALARS {} float\n'.format(key.replace(" ", "_")))
                f.write('LOOKUP_TABLE default\n')
                for item in field.reshape(-1):
                    f.write('    {}\n'.format(item))
                f.write('\n')

            # VECTORS (n1 by n2 by n3 by 3)
            elif(len(field.shape) == 4 and field.shape[3] == 3):
                print('not flatten vector')
                f.write('VECTORS {} float\n'.format(key.replace(" ", "_")))
                for item in field.reshape((field.shape[0] * field.shape[1] * field.shape[2], field.shape[3])):
                    f.write('    {} {} {}\n'.format(*reversed(item)))
                f.write('\n')

            # TENSORS (n1 by n2 by n3 by 3 by 3)
            elif(len(field.shape) == 5 and field.shape[3] * field.shape[4] == 9):
                print('not flatten tensor')
                f.write('TENSORS {} float\n'.format(key.replace(" ", "_")))
                for item in field.reshape((field.shape[0] * field.shape[1] * field.shape[2], field.shape[3] * field.shape[4])):
                    f.write('    {} {} {}\n    {} {} {}\n    {} {} {}\n\n'.format(*reversed(item)))
                f.write('\n')
                
            else:
                print("_writeFieldInVtk(): I'm in an unknown condition!")
                
def json_rand_dictionary(Ncolors, namecmap, dirsave='', first_color_black=True):
    """
    Save a json random colormap to be used with ParaView or Tomviz.
    
    :param Ncolors: Number of labels (size of colormap)
    :type Ncolors: int
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :type type: str
    :param first_color_black: Option to use first color as black, True or False
    :type first_color_black: Bool
    :return: None
    """    
    
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np
    import json
    
    randHSVcolors = [(np.random.uniform(low=0.0, high=1), np.random.uniform(low=0.2, high=1), np.random.uniform(low=0.9, high=1)) for i in range(Ncolors)]
    
    randRGBcolors = []
    for i in range(len(randHSVcolors)):
        RGBcolors = colorsys.hsv_to_rgb(randHSVcolors[i][0], randHSVcolors[i][1], randHSVcolors[i][2])
        
        x0 = i/Ncolors
        if i >0:
            x0 = i/Ncolors#+0.001
        x1 = (i+1)/Ncolors
        
        randRGBcolors.append(x0)
        randRGBcolors.append(RGBcolors[0]); randRGBcolors.append(RGBcolors[1]); randRGBcolors.append(RGBcolors[2])
        randRGBcolors.append(x1)
        randRGBcolors.append(RGBcolors[0]); randRGBcolors.append(RGBcolors[1]); randRGBcolors.append(RGBcolors[2])
    
    if first_color_black == True:
        randRGBcolors[1:4] = [0,0,0]
        randRGBcolors[5:8] = [0,0,0]
    
    
    json_cmap = [
	{
		"ColorSpace" : "HSV",
		"Name" : namecmap,
		"RGBPoints" : randRGBcolors
    }
    ]
    
    with open(dirsave+namecmap, "w") as outfile:
        json.dump(json_cmap, outfile)