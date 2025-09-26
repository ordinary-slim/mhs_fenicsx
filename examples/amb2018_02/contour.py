from paraview.simple import *
import argparse


def get_dims_meltpool(file_name):
    data_set_name = file_name.rstrip("/")
    data_set_name = data_set_name.split("/")[-1]
    data_set_name = data_set_name.rstrip(".bp")
    paraview.simple._DisableFirstRenderCameraReset()

    # create a new 'VTX Reader'
    data_set = ADIOS2VTXReader(registrationName='data_set.bp', FileName=file_name)
    data_set.UpdatePipeline()
    try:
        target_time = data_set.TimestepValues[-1]
    except TypeError:
        target_time = data_set.TimestepValues

    SetActiveSource(data_set)

    # create a new 'Contour'
    contour1 = Contour(registrationName='Contour1', Input=data_set)
    contour1.ContourBy = ['POINTS', 'uh']
    contour1.Isosurfaces = [1290.0]

    contour1.UpdatePipeline(time=target_time)
    bounds = contour1.GetDataInformation().GetBounds()

    L = 1e3*(bounds[1] - bounds[0])
    W = 1e3*(bounds[3] - bounds[2])
    T = 1e3*(bounds[5] - bounds[4])
    print(f"{L}, {W}, {T}")
    return L, W, T 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    args = parser.parse_args()
    get_dims_meltpool(args.data)
