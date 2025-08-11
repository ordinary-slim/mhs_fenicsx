import os
from paraview.simple import *
import argparse
import pandas as pd

def extract_midline_smsbp(dataset_file):
    paraview.simple._DisableFirstRenderCameraReset()
    dataset_name = os.path.basename(dataset_file).split('.')[0]
    dataset_folder = os.path.dirname(dataset_file)

    # create a new 'ADIOS2VTXReader'
    dataset_with_ghosts = ADIOS2VTXReader(registrationName='sms.bp', FileName=dataset_file)

    time = dataset_with_ghosts.TimestepValues[-1]

    dataset = RemoveGhostInformation(registrationName='RemoveGhostInformation1', Input=dataset_with_ghosts)
    dataset = CleantoGrid(registrationName='CleantoGrid1', Input=dataset)

    # get animation scene
    animationScene1 = GetAnimationScene()

    # update animation scene based on data timesteps
    animationScene1.UpdateAnimationUsingDataTimeSteps()

    # get active view
    renderView2 = GetActiveViewOrCreate('RenderView')

    # show data in view
    datasetDisplay = Show(dataset, renderView2, 'UnstructuredGridRepresentation')

    # trace defaults for the display properties.
    datasetDisplay.Representation = 'Surface'

    # reset view to fit data
    renderView2.ResetCamera(False, 0.9)

    #changing interaction mode based on data extents
    renderView2.Set(
        CameraPosition=[0.0, 0.0, 3.35],
        CameraFocalPoint=[0.0, 0.0, 0.0],
    )

    # get the material library
    materialLibrary1 = GetMaterialLibrary()

    # update the view to ensure updated data information
    renderView2.Update()

    # create a new 'Plot Over Line'
    plotOverLine1 = PlotOverLine(registrationName='PlotOverLine1', Input=dataset)
    

    # Properties modified on plotOverLine1
    plotOverLine1.Set(
        SamplingPattern = 'Sample At Cell Boundaries',
        Point1=[-1.0, 0.0, 0.0],
        Point2=[+1.0, 0.0, 0.0],
    )

    plotOverLine1.UpdatePipeline(time=time)

    # save data
    target_csv = os.path.join(dataset_folder, f'{dataset_name}-midline.csv')
    SaveData(target_csv, proxy=plotOverLine1, ChooseArraysToWrite=1, PointDataArrays=['uh'])
    df = pd.read_csv(target_csv)
    df = df.drop(columns=['Points:1', 'Points:2'])
    df = df.rename(columns={'Points:0': 'x'})
    df = df[['x', 'uh']]
    df = df.dropna(subset=['uh']).drop_duplicates(subset=['x'])
    df.to_csv(target_csv, index=False)
    print(f"Midline data saved to {target_csv}")

def extract_midline_sspvd(substep_folder):
    paraview.simple._DisableFirstRenderCameraReset()
    dataset_name = os.path.dirname(substep_folder)
    dataset_name = dataset_name.split("post_")[-1]
    dataset_name = dataset_name.split('_substeps')[0]
    slow_pvd = f"{substep_folder}/{dataset_name}.pvd"
    micro_iters_pvd = f"{substep_folder}/{dataset_name}_micro_iters.pvd"
    # create a new 'PVD Reader'
    micro_iters_dataset = PVDReader(registrationName='micro_iters.pvd', FileName= micro_iters_pvd)
    slow_dataset = PVDReader(registrationName='slow.pvd', FileName= slow_pvd)


    extension = {micro_iters_dataset: 'micro_iters', slow_dataset: 'slow'}

    for dataset in [micro_iters_dataset, slow_dataset]:
        threshold1 = Threshold(Input=dataset)

        threshold1.Set(
            Scalars=['CELLS', 'active_els'],
            UpperThreshold=0.1,
            ThresholdMethod='Above Upper Threshold',
        )

        removeGhostInformation1 = RemoveGhostInformation(Input=threshold1)

        # create a new 'Plot Over Line'
        plotOverLine1 = PlotOverLine(Input=removeGhostInformation1)

        # Properties modified on plotOverLine1
        plotOverLine1.Set(
            SamplingPattern = 'Sample At Cell Boundaries',
            Point1=[-1.0, 0.0, 0.0],
            Point2=[+1.0, 0.0, 0.0],
        )

        plotOverLine1.UpdatePipeline(time=1.0)

        # save data
        target_csv = os.path.join(substep_folder, f'{dataset_name}-{extension[dataset]}-midline.csv')
        SaveData(target_csv, proxy=plotOverLine1, ChooseArraysToWrite=1, PointDataArrays=['uh'])
        df = pd.read_csv(target_csv)
        df = df.drop(columns=['Points:1', 'Points:2'])
        df = df.rename(columns={'Points:0': 'x'})
        df = df[['x', 'uh']]
        df = df.dropna(subset=['uh']).drop_duplicates(subset=['x'])
        df.to_csv(target_csv, index=False)
        print(f"Midline data saved to {target_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    args = parser.parse_args()
    if args.data.endswith('.pvd'):
        extract_midline_smsbp(args.data)
    else:
        extract_midline_sspvd(args.data)
