#!/bin/bash
# SLURM job array for the PD Phase 5 parameter sweep on BSC MareNostrum 5.
#
# Each array task runs one grid point of pd_phase5_hpc_sweep.py and writes a
# small JSON file; a final aggregation step merges them into one CSV.
#
# Submit (the array upper bound must be grid_size - 1; check it first):
#     N=$(python pd_phase5_hpc_sweep.py --list | awk '/total grid/{print $NF}')
#     sbatch --array=0-$((N-1))%50 slurm/pd_phase5_sweep.sbatch
# Then, once the array finishes:
#     python pd_phase5_hpc_sweep.py --aggregate --outdir sweep_out
#
# Notes:
#   * --array=...%50 caps concurrent tasks at 50 (tune to your allocation).
#   * Adjust --account/--qos/--partition to your MareNostrum 5 project.
#   * The mean-field backend is single-core per task; the NEST backend
#     (once implemented) would request multiple cores / a GPU per task and
#     drop the array in favour of MPI ranks per simulation.

#SBATCH --job-name=pd_sweep
#SBATCH --output=logs/pd_sweep_%A_%a.out
#SBATCH --error=logs/pd_sweep_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
###SBATCH --account=<your_bsc_project>
###SBATCH --qos=gp_bscls
###SBATCH --partition=gpp          # general-purpose (Sapphire Rapids)

set -euo pipefail

mkdir -p logs sweep_out

# --- environment -----------------------------------------------------------
# On MareNostrum 5, load a Python module and/or activate a venv with numpy
# (and NEST for the spiking backend). Adjust to your site setup.
# module load python/3.12
# source "$HOME/pd-venv/bin/activate"

OUTDIR="${OUTDIR:-sweep_out}"
BACKEND="${BACKEND:-meanfield}"
TMAX="${TMAX:-2000}"

echo "host=$(hostname) task=${SLURM_ARRAY_TASK_ID:-0} backend=${BACKEND}"

srun python pd_phase5_hpc_sweep.py \
    --run "${SLURM_ARRAY_TASK_ID:-0}" \
    --outdir "${OUTDIR}" \
    --backend "${BACKEND}" \
    --t-max "${TMAX}"
