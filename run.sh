#!/bin/bash -l
#SBATCH --job-name=default
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH -e stderr.txt
#SBATCH -o stdout.txt
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --partition=k2-sandbox
#SBATCH --mem-per-cpu=4G
#SBATCH --ntasks=20
#SBATCH --time=00:30:00
#SBATCH --mail-user=tmoore11@qub.ac.uk
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/users/40178189/myjob.$SLURM_JOBID
module load mpi/intel-mpi
conda activate SNIPER
rm text.txt
mpirun -np 20 python SNIPER.py > text.txt
##################################################

