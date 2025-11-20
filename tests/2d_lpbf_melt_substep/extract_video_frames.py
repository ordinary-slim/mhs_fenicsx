from paraview.simple import *
from mpi4py import MPI
import os
import argparse
import math

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def threshold(ds):
    idx_active = 1.0
    idx_metal = 2.0
    ta = Threshold(Input=ds)
    ta.Set(
        Scalars=['CELLS', 'active_els'],
        UpperThreshold=idx_active - 0.5,
        ThresholdMethod='Above Upper Threshold',
    )
    tm = Threshold(Input=ta)
    tm.Set(
        Scalars=['CELLS', 'material_id'],
        UpperThreshold=idx_metal - 0.5,
        ThresholdMethod='Above Upper Threshold',
    )
    return tm

def initialize_display(problem_display):
    problem_display.Representation = 'Surface With Edges'
    problem_display.EdgeColor = [0.0, 0.0, 0.0]
    problem_display.Opacity = 1.0
    problem_display.EdgeOpacity = 0.7

def set_opacity_ps_display(display, time):
    is_macro_step = math.isclose(round(time), time, abs_tol=1e-6)
    if is_macro_step:
        display.Opacity = 1.0
        display.EdgeOpacity = 0.7
    else:
        display.Opacity = 0.6
        display.EdgeOpacity = 0.6

def set_color_map(array_name):
    fLUT = GetColorTransferFunction(array_name)
    fLUT.Set(
        RGBPoints=GenerateRGBPoints(
            range_min=25,
            range_max=2000,
        ),
        ScalarRangeInitialized=1.0,
    )
    fLUT.ApplyPreset('Rainbow Uniform', True)

def get_screenshots(ps_file, pf_file, pm_file=None):
    paraview.simple._DisableFirstRenderCameraReset()

    ps = ADIOS2VTXReader(registrationName='ps.bp', FileName=ps_file)
    pf = ADIOS2VTXReader(registrationName='pf.bp', FileName=pf_file)
    plist = [ps, pf]
    fplist = [pf]
    if pm_file is not None:
        pm = ADIOS2VTXReader(registrationName='pm.bp', FileName=pm_file)
        plist.append(pm)
        fplist.append(pm)

    # Create a render view and show both datasets
    view = GetActiveViewOrCreate('RenderView')
    viewSize = 1425, 537
    view.Set(
        InteractionMode='2D',
        CameraPosition=[0.0029499615919066596, -0.4503383813138502, 10.05],
        CameraFocalPoint=[0.0029499615919066596, -0.4503383813138502, 0.0],
        CameraParallelScale=0.577385802710209,
    )
    view.OrientationAxesVisibility = 0
    view.ViewSize = viewSize

    tplist = [threshold(p) for p in plist]

    ps_display = Show(tplist[0], view)
    fp_displays = [Show(tplist[i], view) for i in range(1, len(tplist))]
    for disp in [ps_display] + fp_displays:
        initialize_display(disp)
    for disp in fp_displays:
        ColorBy(disp, ('POINTS', 'uh'))
    set_color_map('uh')
    ColorBy(ps_display, ('POINTS', 'net_ext_sol'))
    set_color_map('net_ext_sol')

    # Folder for screenshots
    dataset_name = ps_file.rstrip("/").split("/")[-1].rstrip(".bp")
    out_dir = f"screenshots_{dataset_name}"
    if rank == 0 and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Common times
    common_times = sorted(t for t in ps.TimestepValues if (t in pf.TimestepValues) and (t in pm.TimestepValues if pm_file is not None else True))

    for i, t in enumerate(common_times):
        # Update pipeline at time t
        for p in tplist:
            p.UpdatePipeline(time=t)
        view.ViewTime = t

        set_opacity_ps_display(ps_display, t)

        Render()

        if rank == 0:
            fname = os.path.join(out_dir, f"screenshot_{i:04d}.png")
            SaveScreenshot(
                fname,
                view,
                TransparentBackground=1,
                ImageResolution=viewSize,
            )

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract video frames from LPBF melt substep data.")
    parser.add_argument('--ps-file', type=str, required=True, help='Path to the slow thermal data file.')
    parser.add_argument('--pf-file', type=str, required=True, help='Path to the fast thermal data file.')
    parser.add_argument('--pm-file', type=str, default=None, help='Path to the advected subdomain data file (optional).')
    args = parser.parse_args()
    get_screenshots(args.ps_file, args.pf_file, args.pm_file)

