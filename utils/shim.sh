#$ -S /bin/bash
#$ -q serial
#$ -m e
#$ -M c.wingfield@lancaster.ac.uk

source /etc/profile
module add anaconda3/2018.12

python "$@"
