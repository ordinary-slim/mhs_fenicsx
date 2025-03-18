import yaml

def write_gcode(params):

    partLen = params["part"]
    layer_thickness = params["layer_thickness"]
    speed = max(params["heat_source"]["initial_speed"])
    gcodeFile = params["path"]

    gcodeLines = []
    # Start in -X
    X = - partLen[0] / 2
    Y = 0.0
    Z = 0.0
    E = 0.0
    max_layers = int(partLen[2] / layer_thickness)
    n_layers = params["n_layers"]
    numLayers = min( n_layers, max_layers )
    
    gcodeLines.append( "G0 F{}".format(speed, X, Y, Z) )
    for ilayer in range(numLayers):
        Z = layer_thickness * (ilayer)
        E += 0.1
        gcodeLines.append( "G0 X{} Y{} Z{:.3f}".format(X, Y, Z) )

        gcodeLines.append( "G4 P{}".format( params["interLayerDelay"] / 2 ) )
        gcodeLines.append( "G4 P{} R1".format( params["interLayerDelay"] / 2 ) )

        X = -X
        gcodeLines.append( "G1 X{} E{:.2f}".format(X, E) )

    with open(gcodeFile, 'w') as gfile:
        gfile.writelines( [line+"\n" for line in gcodeLines] )

if __name__=="__main__":
    write_gcode()
