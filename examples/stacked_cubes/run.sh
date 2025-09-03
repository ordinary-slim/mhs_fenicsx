descriptor=$1
numProcs=$2
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
source /data0/home/mslimani/bin/FEniCSx/load_setup_functions.sh
setup_fenicsx_env
# mpirun -n ${numProcs} python3 main.py -r -i $INPUT_FILE -d $descriptor >> $printFile
# mpirun -n ${numProcs} python3 main.py -ss -i $INPUT_FILE -d $descriptor >> $printFile
# mpirun -n ${numProcs} python3 main.py -css -d mono_${descriptor}_mono -i $MONOLITHIC_INPUT_FILE >> $printFile
mpirun -n ${numProcs} python3 main.py -css -d stag_${descriptor} -i $INPUT_FILE >> $printFile
# mpirun -n ${numProcs} python3 main.py -sms -i $INPUT_FILE -d $descriptor >> $printFile
find . -maxdepth 1 -name '*profiling*.txt' -exec mv -t "$PROFILINGS_DIR" {} +
