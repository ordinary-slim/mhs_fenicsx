import os
from paraview.simple import *
import argparse
import pandas as pd
import numpy as np

def cleanup_csv(target_csv):
    df = pd.read_csv(target_csv)
    df = df.drop(columns=['Points:1', 'Points:2'])
    df = df.rename(columns={'Points:0': 'x'})
    df = df.dropna(subset=['uh']).drop_duplicates(subset=['x'])
    df.to_csv(target_csv, index=False)

def extract_midline_smsbp(dataset_file):
    paraview.simple._DisableFirstRenderCameraReset()
    dataset_name = os.path.basename(dataset_file.rstrip("/")).split('.')[0]
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

    cleanup_csv(target_csv)
    print(f"Midline data saved to {target_csv}")

def extract_midline_sspvd(substep_folder, times=[1.0]):
    paraview.simple._DisableFirstRenderCameraReset()
    dataset_name = os.path.dirname(substep_folder)
    if not(dataset_name):
        dataset_name = os.path.basename(substep_folder)
    dataset_name = dataset_name.split("post_")[-1]
    dataset_name = dataset_name.split('_substeps')[0]
    slow_pvd = f"{substep_folder}/{dataset_name}.pvd"
    micro_iters_pvd = f"{substep_folder}/{dataset_name}_micro_iters.pvd"
    # create a new 'PVD Reader'
    micro_iters_dataset = PVDReader(registrationName='micro_iters.pvd', FileName= micro_iters_pvd)
    slow_dataset = PVDReader(registrationName='slow.pvd', FileName= slow_pvd)

    extension = {micro_iters_dataset: 'micro_iters', slow_dataset: 'slow'}
    mode = "sms" if 'sms' in dataset_name else "ss"
    upper_threshold_activation = 0.1 * (mode == "ss")

    if isinstance(times, (str)):
        times = [float(t) for t in times.split(',')]
        if times[0] == -1.0:
            times = micro_iters_dataset.TimestepValues

    for time in times:
        df = {micro_iters_dataset: None, slow_dataset: None}
        for dataset in [slow_dataset, micro_iters_dataset]:
            if mode == "sms" and dataset == slow_dataset:
                continue
            threshold1 = Threshold(Input=dataset)

            threshold1.Set(
                Scalars=['CELLS', 'active_els'],
                UpperThreshold=upper_threshold_activation,
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

            plotOverLine1.UpdatePipeline(time=time)

            # save data
            target_csv = os.path.join(substep_folder, f'{dataset_name}-{extension[dataset]}-midline.csv')
            SaveData(target_csv, proxy=plotOverLine1, ChooseArraysToWrite=1, PointDataArrays=['uh'])
            df[dataset] = pd.read_csv(target_csv)
            df[dataset] = df[dataset].drop(columns=['Points:1', 'Points:2'])
            df[dataset] = df[dataset].rename(columns={'Points:0': 'x'})
            df[dataset] = df[dataset][['x', 'uh']]
            df[dataset] = df[dataset].dropna(subset=['uh']).drop_duplicates(subset=['x'])
            df[dataset] = df[dataset].sort_values("x").reset_index(drop=True)
            if dataset == slow_dataset:
                dx = df[dataset]["x"].diff()
                gap_indices = dx[dx > 0.1].index
                assert(len(gap_indices) == 1)
                gap_idx = gap_indices[0]
        if mode == "ss":
            sides_left  = df[slow_dataset].iloc[:gap_idx].copy()
            sides_right = df[slow_dataset].iloc[gap_idx:].copy()
            x_left  = sides_left["x"].iloc[-1]
            x_right = sides_right["x"].iloc[0]
            nan_row_left = pd.DataFrame({'x': [x_left], 'uh': [np.nan]})
            nan_row_right = pd.DataFrame({'x': [x_right], 'uh': [np.nan]})
            df = pd.concat([sides_left, nan_row_left, df[micro_iters_dataset], nan_row_right, sides_right], ignore_index=True)
        else:
            df = df[micro_iters_dataset]

        csvname = f'{dataset_name}-midline-t{time:g}'
        csvname = csvname.replace(".", "_") + ".csv"
        target_csv = os.path.join(substep_folder, csvname)
        df.to_csv(target_csv, index=False, na_rep='NaN')
        print(f"Midline data saved to {target_csv}")

def wrong_interpolation_csvs(substep_folder, predictor_bp):
    paraview.simple._DisableFirstRenderCameraReset()
    dataset_name = os.path.dirname(substep_folder)
    dataset_name = dataset_name.split("post_")[-1]
    dataset_name = dataset_name.split('_substeps')[0]

    predictor_name = os.path.basename(predictor_bp.rstrip("/")).split('.')[0]
    predictor_folder = os.path.dirname(predictor_bp)

    micro_iters_pvd = f"{substep_folder}/{dataset_name}_micro_iters.pvd"
    # create a new 'PVD Reader'
    micro_iters_dataset = PVDReader(registrationName='micro_iters.pvd', FileName= micro_iters_pvd)

    predictor_ds_raw = ADIOS2VTXReader(registrationName='sms_predictor.bp', FileName=predictor_bp)
    predictor_ds = RemoveGhostInformation(Input=predictor_ds_raw)

    threshold1 = Threshold(Input=micro_iters_dataset)
    threshold1.Set(
        Scalars=['CELLS', 'active_els'],
        UpperThreshold=0.1,
        ThresholdMethod='Above Upper Threshold',
    )

    active_ds = RemoveGhostInformation(Input=threshold1)
    ds = RemoveGhostInformation(Input=micro_iters_dataset)

    active_plot = PlotOverLine(Input=active_ds)
    plot = PlotOverLine(Input=ds)
    predictor_plot = PlotOverLine(Input=predictor_ds)

    for p in [active_plot, plot, predictor_plot]:
        p.Set(
            SamplingPattern = 'Sample At Cell Boundaries',
            Point1=[-1.0, 0.0, 0.0],
            Point2=[+1.0, 0.0, 0.0],
        )

    ## FROM PLOT, WE WANT UH_N AND UH AT TIME 0, AND UH AT TIME 1.0

    def get_target_csv(time, sol=True):
        time_str = str(time).replace(".", "_")
        if sol:
            folder = substep_folder
            dsname = dataset_name
        else:
            # predictor
            folder = predictor_folder
            dsname = predictor_name
        csv_name = f'{dsname}-midline_t{time_str}.csv'
        csv_name = csv_name.replace("#", "")
        return os.path.join(folder, csv_name)

    def write_csv(p, time, arrays, sol=True):
        p.UpdatePipeline(time=time)
        # save data
        target_csv = get_target_csv(time, sol=sol)
        SaveData(target_csv,
                 proxy=p,
                 ChooseArraysToWrite=1,
                 PointDataArrays=arrays)
        cleanup_csv(target_csv)
        print(f"Wrote {target_csv} for time {time}")

    write_csv(plot, 0.0, ['uh', 'uh_n'])
    write_csv(plot, 1.0, ['uh'])

    ## FROM ACTIVE_PLOT, WE WANT UH at intermediate times
    ## SAME FROM PREDICTOR_PLOT
    for t in micro_iters_dataset.TimestepValues:
        if np.isclose(t, 0.0) or np.isclose(t, 1.0):
            continue
        write_csv(active_plot, t, ['uh'], sol=True)
        tpred = predictor_ds_raw.TimestepValues[np.argmin([abs(tp - t) for tp in predictor_ds_raw.TimestepValues])]
        write_csv(predictor_plot, tpred, ['uh'], sol=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', required=True, help='Path to the dataset file (PVD or BP)')
    parser.add_argument('-b', '--data-bis', default=None, help='Path to the second dataset file (for wrong interpolation CSVs)')
    parser.add_argument('-t', '--time', default=[1.0])
    arguments = parser.parse_args()

    if arguments.data_bis is not None:
        wrong_interpolation_csvs(arguments.data, arguments.data_bis)
    elif arguments.data.endswith('.bp'):
        extract_midline_smsbp(arguments.data)
    elif "substeps_tstep" in arguments.data:
        extract_midline_sspvd(arguments.data, times=arguments.time)
