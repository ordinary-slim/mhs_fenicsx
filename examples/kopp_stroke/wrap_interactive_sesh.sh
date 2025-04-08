#!/bin/bash
jobName=interactice_session
partition=R640
printFile=$jobName.print
nps=104
srun --partition=$partition --job-name="kopp_strokes" --ntasks=$nps --mem-per-cpu=4G --time=10:00:00  --pty bash -i
