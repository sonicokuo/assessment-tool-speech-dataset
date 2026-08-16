#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=5
#SBATCH -t 3:00:00
#SBATCH -A cis260125p
#SBATCH -J trivnulls
SH=/ocean/projects/cis260125p/shared
cd $SH/repo_verify
$SH/envs/project/bin/python -u $SH/trivial_nulls.py || exit 1
echo "TRIVNULLS_DONE"
