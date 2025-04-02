#!/bin/bash
runType=$1
descriptor=$2
jobName="kopp_stroke_${runType}_${descriptor}"
partition=R640
printFile=$jobName.print
nps=10
echo $jobName
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name="$jobName"
#SBATCH --partition=$partition
#SBATCH --ntasks=$nps
#SBATCH --mem-per-cpu=4G
#SBATCH --time=2-0
#SBATCH --output=$jobName.out
#SBATCH --error=$jobName.err
source /data0/home/mslimani/bin/FEniCSx/load_setup_functions.sh
setup_fenicsx_env
mpirun -n $nps python3 main.py -$runType -d $descriptor
EOT
