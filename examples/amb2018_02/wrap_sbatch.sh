#!/bin/bash
nps=$1
descriptor=$2
partition=$3
jobName="kopp_stroke_${descriptor}"
printFile=$jobName.print
echo $jobName
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name="$jobName"
#SBATCH --partition=$partition
#SBATCH --nodes=1
#SBATCH --ntasks=$nps
#SBATCH --mem=100G
#SBATCH --time=2-0
#SBATCH --output=$jobName.out
#SBATCH --error=$jobName.err
source /data0/home/mslimani/bin/FEniCSx/load_setup_functions.sh
setup_fenicsx_env
mpirun -n $nps python3 calibration.py -l
EOT
