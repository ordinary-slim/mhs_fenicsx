#!/bin/bash
nps=$1
descriptor=$2
partition=$3
jobName="stacked_cubes_${descriptor}"
printFile=$jobName.print
PROFILINGS_DIR="profilings_${descriptor}"
mkdir ${PROFILINGS_DIR}
INPUT_FILE="tmp_${jobName}.yaml"
MONOLITHIC_INPUT_FILE="tmp_${jobName}_mono.yaml"
cp input.yaml $INPUT_FILE
sed -i "s/\<monolithic\>/staggered/" $INPUT_FILE
cp input.yaml $MONOLITHIC_INPUT_FILE
sed -i "s/\<staggered\>/monolithic/" $MONOLITHIC_INPUT_FILE
echo $jobName
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name="$jobName"
#SBATCH --partition=$partition
#SBATCH --nodes=1
#SBATCH --ntasks=$nps
#SBATCH --mem=200G
#SBATCH --time=2-0
#SBATCH --output=./${PROFILINGS_DIR}/${jobName}.out
#SBATCH --error=./${PROFILINGS_DIR}/${jobName}.err
source /data0/home/mslimani/bin/FEniCSx/load_setup_functions.sh
setup_fenicsx_env
mpirun -n $nps python3 main.py -r -i $INPUT_FILE
mpirun -n $nps python3 main.py -ss -i $INPUT_FILE
mpirun -n $nps python3 main.py -css -d monolithic -i $MONOLITHIC_INPUT_FILE
mpirun -n $nps python3 main.py -css -d staggered -i $INPUT_FILE
mpirun -n $nps python3 main.py -sms -i $INPUT_FILE
# mpirun -n $nps python3 main.py -csms -d monolithic -i $MONOLITHIC_INPUT_FILE
sleep 10 # Wait for io ops to finish
find . -maxdepth 1 -name '*profiling*.txt' -exec mv -t "$PROFILINGS_DIR" {} +
EOT
