#!/bin/bash
jobName=interactice_session
partition=R640
printFile=$jobName.print
nps=52
srun --partition=$partition --job-name="kopp_strokes" --ntasks=$nps --mem-per-cpu=4G --time=24:00:00  --pty bash -i
