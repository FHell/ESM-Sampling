#!/bin/bash
# model256_era5.h5 from the cluster's ERA5 folder (no internet needed; a compute node reads
# /p/projects/climate_data_central directly). Needs `data.sbatch pypsa256` first. Preemptible:
# a requeued job keeps the finished monthly cut-outs and continues.
#   sbatch hpc/jobs/era5.sbatch [years, default 2003,2010,2013,2017]
#SBATCH --account=icone
#SBATCH --partition=priority
#SBATCH --qos=standby
#SBATCH --requeue
#SBATCH --job-name=esm-era5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --chdir=/p/tmp/hellmann/esm
#SBATCH --output=/p/tmp/hellmann/esm/logs/%x-%j.out
ESM_PROJ=${ESM_PROJ:-/p/projects/coen/esm}
export UV_OFFLINE=1
bash "$ESM_PROJ/ESM-Sampling/hpc/prepare_data.sh" era5 "${1:-}"
