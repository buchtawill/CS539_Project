#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=32g
#SBATCH -J "Grayscale image pair training"
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH -C H100|A100
#SBATCH --mail-type=BEGIN,FAIL,END

module purge
module load slurm cuda12.1 #python/3.12.4

echo "INFO [run.sh] Starting execution"

#source /home/jwbuchta/CS539_Project/Autoencoder/venv_autoencoder/bin/activate
#which $HOME/CS539_Project/Autoencoder/venv_autoencoder/bin/python
$HOME/CS539_Project/Autoencoder/venv_autoencoder/bin/python colorizer.py

#sleep 600

echo "INFO [run.sh] Script finished"
