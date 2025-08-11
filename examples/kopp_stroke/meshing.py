import numpy as np
import gmsh
from dolfinx.io.gmshio import model_to_mesh
from dolfinx.mesh import GhostMode, create_cell_partitioner
from mpi4py import MPI
import argparse

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def boundary_layer_progression(fine_el_size, coarsening_factor, len_extrusion):
    Sn = (len_extrusion / fine_el_size)
    num_layers = np.rint(np.emath.logn(coarsening_factor, -(len_extrusion/fine_el_size*(1 - coarsening_factor) - coarsening_factor))).astype(int)
    heights = coarsening_factor**np.arange(1, num_layers+1)
    num_elements_per_layer = [1]*num_layers

    # format heights to cumsum
    heights = np.cumsum( heights )
    heights /= heights[-1]

    return num_elements_per_layer, heights

def get_mesh(params, symmetry=False):
    gmsh.initialize()
    if rank == 0:
        # PARAMS
        part_lens = np.array(params["part"])
        substrate_lens = np.array(params["substrate"])
        radius = params["source_terms"][0]["radius"]
        els_per_radius = params["els_per_radius"]
        fine_el_size = radius / els_per_radius
        adim_fine_pads = params["adim_fine_pads"]
        nboun_layers = [np.rint(pad * els_per_radius).astype(int) \
            for pad in adim_fine_pads]
        coarsening_factor = params["coarsening_factor"]
        # Adjust lengths so that they are multiples of fine_el_size
        for lens in [part_lens, substrate_lens]:
            lens[:] = (np.ceil(lens / fine_el_size) * fine_el_size)[:]
        half_lens_part = np.array(part_lens) / 2
        half_lens_substrate = np.array(substrate_lens) / 2
        # Bot surface part
        llcorner = np.array([-half_lens_part[0], -half_lens_part[1], 0.0])
        lrcorner = np.array([-half_lens_part[0], +half_lens_part[1], 0.0])
        urcorner = np.array([+half_lens_part[0], +half_lens_part[1], 0.0])
        ulcorner = np.array([+half_lens_part[0], -half_lens_part[1], 0.0])
        if symmetry:
            llcorner[1] = 0.0
            ulcorner[1] = 0.0
        gmsh.model.geo.addPoint(*llcorner, tag=1)
        gmsh.model.geo.addPoint(*lrcorner, tag=2)
        gmsh.model.geo.addPoint(*urcorner, tag=3)
        gmsh.model.geo.addPoint(*ulcorner, tag=4)
        linesBottomSurfacePart = []
        linesBottomSurfacePart.append(gmsh.model.geo.addLine(1, 2, tag=1))
        linesBottomSurfacePart.append(gmsh.model.geo.addLine(2, 3, tag=2))
        linesBottomSurfacePart.append(gmsh.model.geo.addLine(3, 4, tag=3))
        linesBottomSurfacePart.append(gmsh.model.geo.addLine(4, 1, tag=4))
        for idx, line in enumerate(linesBottomSurfacePart):
            if (idx % 2) != 0:
                pL = part_lens[0]
            else:
                pL = part_lens[1]
                if symmetry:
                    pL /= 2.0
            numEls = pL / fine_el_size
            numEls = np.rint(numEls).astype(int)
            gmsh.model.geo.mesh.setTransfiniteCurve(line, numEls+1 )
        curveLoopBotSurfacePart = gmsh.model.geo.addCurveLoop( linesBottomSurfacePart, 1 )
        botSurfacePart = gmsh.model.geo.addPlaneSurface([curveLoopBotSurfacePart], 1)
        gmsh.model.geo.mesh.setTransfiniteSurface(botSurfacePart)
        gmsh.model.geo.mesh.setRecombine(2,botSurfacePart) 

        # Substrate extrusions
        ## Substrate extrusions Z
        ## Uniform extrusion
        nboun_layers_z = nboun_layers[2]
        lenUniformBotExtrusion = nboun_layers_z*fine_el_size
        uniformBotExtrusion = gmsh.model.geo.extrude([(2, botSurfacePart)], 0, 0, -lenUniformBotExtrusion, numElements =[nboun_layers_z], recombine= True)
        midBotSurface, _, xSideMidBot1, ySideMidBot1, xSideMidBot2, ySideMidBot2 = uniformBotExtrusion
        ## Coarsening
        lenCoarseBotExtrusion = substrate_lens[2] - nboun_layers_z*fine_el_size
        nElements, heights = boundary_layer_progression(fine_el_size, coarsening_factor, lenCoarseBotExtrusion)
        coarseBotExtrusion = gmsh.model.geo.extrude([(2, midBotSurface[1])], 0, 0, -lenCoarseBotExtrusion, numElements = nElements, heights= heights, recombine= True)
        botBotSurface, _, xSideBotBot1, ySideBotBot1, xSideBotBot2, ySideBotBot2 = coarseBotExtrusion
        # Substrate extrusions Y
        ## Uniform extrusions
        uniformYExtrusions = []
        nboun_layers_y = nboun_layers[1]
        lenUniformYExtrusion = nboun_layers_y*fine_el_size
        for idx, surface in enumerate([ySideMidBot1, ySideMidBot2, ySideBotBot1, ySideBotBot2]):
            if symmetry:
                if idx % 2 == 1:
                    uniformYExtrusions.append(None)
                    continue
            uniformYExtrusions.append(gmsh.model.geo.extrude([(2, surface[1])], 0.0, np.power(-1, idx)*lenUniformYExtrusion, 0.0, numElements =[nboun_layers_y], recombine=True))

        ## Coarse extrusions
        coarseYExtrusions = []
        lenCoarseYExtrusion = (substrate_lens[1] - part_lens[1])/2 - nboun_layers_y*fine_el_size
        nElements, heights = boundary_layer_progression(fine_el_size, coarsening_factor, lenCoarseYExtrusion)
        for idx, extrusion in enumerate(uniformYExtrusions):
            if extrusion is None:
                coarseYExtrusions.append(None)
                continue
            dy = np.power(-1, idx)*lenCoarseYExtrusion
            tagSurface = extrusion[0][1]
            coarseYExtrusions.append( gmsh.model.geo.extrude([(2, tagSurface)], 0.0, dy, 0.0, numElements = nElements, heights= heights, recombine= True) )

        # Substrate extrusions X
        ## Uniform extrusions
        positiveUniformExtrusionsX = []
        negativeUniformExtrusionsX = []
        surfacesXPlus = []
        surfacesXMinus = []
        # Collect all X surfaces
        for extrusion in [uniformBotExtrusion, coarseBotExtrusion]:
            surfacesXMinus.append( extrusion[2][1] )
            surfacesXPlus.append( extrusion[4][1] )
        for idx, extrusion in enumerate(uniformYExtrusions + coarseYExtrusions):
            if extrusion is None:
                continue
            if (idx%2 == 0):
                surfacesXPlus.append( extrusion[3][1] )
                surfacesXMinus.append( extrusion[5][1] )
            else:
                surfacesXMinus.append( extrusion[3][1] )
                surfacesXPlus.append( extrusion[5][1] )
        nboun_layers_x = nboun_layers[0]
        extrusionLen = nboun_layers_x*fine_el_size
        numElements = extrusionLen / fine_el_size
        for surface in surfacesXPlus:
            positiveUniformExtrusionsX.append( gmsh.model.geo.extrude([(2, surface)], extrusionLen, 0.0, 0.0, numElements =[numElements], recombine= True ) )
        for surface in surfacesXMinus:
            negativeUniformExtrusionsX.append( gmsh.model.geo.extrude([(2, surface)], -extrusionLen, 0.0, 0.0, numElements =[numElements], recombine= True ) )
        ## Coarse extrusions
        extrusionLen = (substrate_lens[0] - part_lens[0])/2 - nboun_layers_x*fine_el_size
        nElements, heights = boundary_layer_progression(fine_el_size, coarsening_factor, extrusionLen)
        positiveCoarseExtrusionsX = []
        negativeCoarseExtrusionsX = []
        for extrusion in positiveUniformExtrusionsX:
            positiveCoarseExtrusionsX.append ( gmsh.model.geo.extrude([extrusion[0]], +extrusionLen, 0.0, 0.0, numElements = nElements, heights = heights,  recombine = True ) )
        for extrusion in negativeUniformExtrusionsX:
            negativeCoarseExtrusionsX.append( gmsh.model.geo.extrude([extrusion[0]], -extrusionLen, 0.0, 0.0, numElements = nElements, heights = heights, recombine = True ) )

        # Top extrusion
        #nelsPartZ = part_lens[2] / fine_el_size
        #substrateTopSurfaces = []
        #substrateTopSurfaces.append( (2, botSurfacePart) )
        #substrateTopSurfaces.append( uniformYExtrusions[0][2] )
        #substrateTopSurfaces.append( uniformYExtrusions[1][2] )
        #substrateTopSurfaces.append( coarseYExtrusions[0][2] )
        #substrateTopSurfaces.append( coarseYExtrusions[1][2] )
        #substrateTopSurfaces.append( positiveUniformExtrusionsX[0][2] )
        #substrateTopSurfaces.append( positiveUniformExtrusionsX[2][-1] )
        #substrateTopSurfaces.append( positiveUniformExtrusionsX[3][3] )
        #substrateTopSurfaces.append( positiveUniformExtrusionsX[6][-1] )
        #substrateTopSurfaces.append( positiveUniformExtrusionsX[7][3] )
        #substrateTopSurfaces.append( positiveCoarseExtrusionsX[0][2] )
        #substrateTopSurfaces.append( positiveCoarseExtrusionsX[2][-1] )
        #substrateTopSurfaces.append( positiveCoarseExtrusionsX[3][3] )
        #substrateTopSurfaces.append( positiveCoarseExtrusionsX[6][-1] )
        #substrateTopSurfaces.append( positiveCoarseExtrusionsX[7][3] )
        ##
        #substrateTopSurfaces.append( negativeUniformExtrusionsX[0][2] )
        #substrateTopSurfaces.append( negativeUniformExtrusionsX[2][3] )
        #substrateTopSurfaces.append( negativeUniformExtrusionsX[3][-1] )
        #substrateTopSurfaces.append( negativeUniformExtrusionsX[6][3] )
        #substrateTopSurfaces.append( negativeUniformExtrusionsX[7][-1] )
        #substrateTopSurfaces.append( negativeCoarseExtrusionsX[0][2] )
        #substrateTopSurfaces.append( negativeCoarseExtrusionsX[2][3] )
        #substrateTopSurfaces.append( negativeCoarseExtrusionsX[3][-1] )
        #substrateTopSurfaces.append( negativeCoarseExtrusionsX[6][3] )
        #substrateTopSurfaces.append( negativeCoarseExtrusionsX[7][-1] )
        #topExtrusions = []
        #for surface in substrateTopSurfaces:
        #    topExtrusions.append( gmsh.model.geo.extrude([surface], 0, 0, +part_lens[2], numElements =[nelsPartZ], recombine= True) )


        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(3)

        volumeTags = []
        for _, tag in gmsh.model.getEntities( 3 ):
            volumeTags.append( tag )
        gmsh.model.addPhysicalGroup(3, volumeTags, tag = 1, name="Domain")

    model = MPI.COMM_WORLD.bcast(gmsh.model, root = 0)
    partitioner = create_cell_partitioner(GhostMode.shared_facet)
    msh_data = model_to_mesh(model, MPI.COMM_WORLD, 0, gdim = 3,partitioner= partitioner)
    msh = msh_data[0]

    gmsh.finalize()
    MPI.COMM_WORLD.barrier()
    return msh

if __name__ == "__main__":
    import yaml
    from dolfinx import io

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', default="input.yaml")

    args = parser.parse_args()

    params_file = args.input_file
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)

    domain = get_mesh(params)
    with io.VTKFile(domain.comm, "mesh.pvd", "wb") as f:
        f.write_mesh(domain)
