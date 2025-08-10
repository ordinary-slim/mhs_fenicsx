#!/bin/bash
nps=$1
runType=$2
descriptor="calibrate_T1nu1"
jobName="kopp_stroke_${runType}_${descriptor}"
partition=$3
printFile=$jobName.print
echo $jobName
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name="$jobName"
#SBATCH --partition=$partition
#SBATCH --nodes=1
#SBATCH --ntasks=$nps
#SBATCH --mem-per-cpu=4G
#SBATCH --time=2-0
#SBATCH --output=$jobName.out
#SBATCH --error=$jobName.err
source /data0/home/mslimani/bin/FEniCSx/load_setup_functions.sh
setup_fenicsx_env
mpirun -n $nps python3 calibration.py -l
EOT
