#$ -S /bin/bash
#$ -q serial
#$ -m e
#$ -M notify@cwcomplex.net

source /etc/profile
module add anaconda3/2018.12

python "$@"
