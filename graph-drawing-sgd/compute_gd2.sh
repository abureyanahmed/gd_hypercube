#!/bin/bash
#SBATCH --account=mstrout
#SBATCH --partition=standard
#SBATCH --ntasks=28
#SBATCH --mem-per-cpu=5GB
#SBATCH --time=8:00:00

module load python/3.6/3.6.5
module load matlab/r2020b

singularity exec ../../mlst_kruskal/steiner_docker.simg python3 run_gd2.py $MAP_FILE $SLURM_ARRAY_TASK_ID > $LOG_FOLDER/output_$SLURM_ARRAY_TASK_ID.dat 2> $LOG_FOLDER/error_$SLURM_ARRAY_TASK_ID.dat


