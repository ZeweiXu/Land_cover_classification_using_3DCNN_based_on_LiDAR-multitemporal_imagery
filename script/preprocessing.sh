#!/bin/tcsh
#SBATCH -n 12
#SBATCH --time=10:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --mail-user=zeweixu2@illinois.edu


 
module purge
module add GNU610
module use /data/cigi/common/cigi-modules
module load anaconda2
source /data/keeling/a/zeweixu2/pointnet2/bin/activate
cd /data/cigi/common/zeweixu2/land_cover_scale_up/script 
python preprocessing.py $1 $2 $3
