#$ -cwd
#$ -q short.qc
#$ -N resnet101
#$ -j y
#$ -o /well/papiez/users/zou393/Documents/DPhil/script/log-resnet101

echo "------------------------------------------------"
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "------------------------------------------------"

man module
module load Python/3.7.2-GCCcore-8.2.0
module load Anaconda3/5.3.0
source /apps/eb/skylake/software/Anaconda3/5.3.0/bin/conda

sleep 60s
python -c 'print("Hello")'


#conda env create -f mimic.yml
#conda activate mimic
#python train.py --config resnet101

