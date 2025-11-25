from paraview.simple import *
from mpi4py import MPI
import os
import argparse
import math
import subprocess

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

max_edge_opacity = 0.3
min_edge_opacity = 0.2

def threshold(ds):
    idx_active = 1.0
    idx_powder = 1.0
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
    tp = Threshold(Input=ta)
    tp.Set(
        Scalars=['CELLS', 'material_id'],
        LowerThreshold=idx_powder + 0.5,
        ThresholdMethod='Below Lower Threshold',
    )
    return tm, tp

def initialize_display(problem_display, is_powder=False):
    if is_powder:
        problem_display.Representation = 'Wireframe'
    else:
        problem_display.Representation = 'Surface With Edges'
    problem_display.EdgeColor = [1.0, 1.0, 1.0]
    problem_display.Opacity = 1.0
    problem_display.EdgeOpacity = max_edge_opacity

def is_macro_step(time):
    return math.isclose(round(time), time, abs_tol=1e-6)

def set_opacity_ps_display(display, time):
    if is_macro_step(time):
        display.Opacity = 1.0
        display.EdgeOpacity = max_edge_opacity
    else:
        display.Opacity = 0.6
        display.EdgeOpacity = min_edge_opacity

def set_color_map(array_name):
    fLUT = GetColorTransferFunction(array_name)
    fLUT.Set(
        RGBPoints=GenerateRGBPoints(
            range_min=25,
            range_max=1650,
        ),
        ScalarRangeInitialized=1.0,
    )
    fLUT.ApplyPreset('Inferno (matplotlib)', True)

def get_screenshots(ps_file, pf_file, pm_file=None, video=False):
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
    viewSize = 1240, 540
    view.Set(
        InteractionMode='2D',
        CameraPosition=[-0.00013824362807540735, -0.40009415420574357, 5.8116313393201295],
        CameraFocalPoint=[-0.00013824362807540735, -0.40009415420574357, 0.0],
        CameraParallelScale=0.6145544469847664,
    )
    view.OrientationAxesVisibility = 0
    view.ViewSize = viewSize
    # find settings proxy
    colorPalette = GetSettingsProxy('ColorPalette')

    # white bg
    colorPalette.Background = [1.0, 1.0, 1.0]

    metal_thresholds, powder_thresholds = zip(*(threshold(p) for p in plist))

    ps_metal_display, ps_powder_display = Show(metal_thresholds[0], view), Show(powder_thresholds[0], view)
    fast_metal_displays = [Show(metal_thresholds[i], view) for i in range(1, len(metal_thresholds))]
    fast_powder_displays = [Show(powder_thresholds[i], view) for i in range(1, len(powder_thresholds))]
    slow_displays = [ps_metal_display, ps_powder_display]
    fast_displays = fast_metal_displays + fast_powder_displays

    for disp in [ps_metal_display] + fast_metal_displays:
        initialize_display(disp)
    for disp in [ps_powder_display] + fast_powder_displays:
        initialize_display(disp, is_powder=True)

    for disp in fast_displays:
        ColorBy(disp, ('POINTS', 'uh'))
    set_color_map('uh')
    for disp in slow_displays:
        ColorBy(disp, ('POINTS', 'net_ext_sol'))
    set_color_map('net_ext_sol')

    # Folder for screenshots
    dataset_name = ps_file.rstrip("/").split("/")[-1].rstrip(".bp")
    out_dir = f"screenshots_{dataset_name}"
    if rank == 0 and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Common times
    common_times = sorted(t for t in ps.TimestepValues if (t in pf.TimestepValues) and (t in pm.TimestepValues if pm_file is not None else True))

    screenshots = []

    annotateTime1 = AnnotateTime(Format=f"")
    annotateTime1Display = Show(annotateTime1, view, 'TextSourceRepresentation')
    annotateTime1Display.Color = [0.0, 0.0, 0.1600061058998108]
    annotateTime1Display.WindowLocation = 'Any Location'
    annotateTime1Display.Position = [0.01, 0.92]
    annotateTime1Display.Justification = 'Left'
    annotateTime1Display.FontFamily = 'File'
    annotateTime1Display.FontFile = '/usr/share/fonts/liberation/LiberationMono-Regular.ttf'

    for i, t in enumerate(common_times):
        # Update pipeline at time t
        annotateTime1.Format = f"Micro-step # {i+1}\nMacro-step # {math.floor(t)}"
        for p in plist:
            p.UpdatePipeline(time=t)
        view.ViewTime = t

        for sdsp in slow_displays:
            set_opacity_ps_display(sdsp, t)

        Render()

        fname = os.path.join(out_dir, f"screenshot_{i:04d}.png")

        if rank == 0:
            SaveScreenshot(
                fname,
                view,
                ImageResolution=viewSize,
            )
        screenshots.append((fname, t))

    if video and rank == 0:
        output_video = f"2dlpbf_{dataset_name}.mp4"
        screenshots_to_video(screenshots, output_video)

def screenshots_to_video(screenshots, output_video):
    # Write screenshots to list.txt file for ffmpeg
    min_dt = None
    min_duration = 0.125
    with open("list.txt", "w") as f:
        for i, (fname, t) in enumerate(screenshots):
            if i == 0:
                min_dt = t
                dt = min_dt
            else:
                dt = t - screenshots[i-1][1]
            if is_macro_step(t):
                duration = 4*min_duration
            else:
                duration = round((dt / min_dt)*min_duration, 2)
            print(t, min_dt, dt, duration)
            f.write(f"file '{fname}'\n")
            f.write(f"outpoint {duration}\n")
        # Repeat last frame to ensure it stays on screen
        f.write(f"file '{screenshots[-1][0]}'\n")
    # Call ffmpeg through subprocess
    subprocess.run([
        "ffmpeg",
        "-f", "concat",
        "-i", "list.txt",
        "-framerate", "1",
        "-c:v", "libx264", "-c:a", "copy", "-shortest", "-r", "30", "-pix_fmt", "yuv420p",
        output_video
    ])

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract video frames from LPBF melt substep data.")
    parser.add_argument('--ps-file', type=str, required=True, help='Path to the slow thermal data file.')
    parser.add_argument('--pf-file', type=str, required=True, help='Path to the fast thermal data file.')
    parser.add_argument('--pm-file', type=str, default=None, help='Path to the advected subdomain data file (optional).')
    parser.add_argument('--video', action='store_true', help='Generate a video from the screenshots.')
    args = parser.parse_args()
    get_screenshots(args.ps_file, args.pf_file, args.pm_file, video=args.video)
