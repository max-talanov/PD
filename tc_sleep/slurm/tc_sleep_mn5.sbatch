#!/bin/bash
# SLURM job for the auditory thalamo-cortical sleep model on BSC MareNostrum 5.
#
# Unlike the PD Phase 5 parameter sweep (a job array), this is a SINGLE long,
# large, bio-plausible simulation run with shared-memory parallelism: one task,
# many NEST threads. Scale up further with --network_auditory_mn5.yaml sizes
# and/or by raising --cpus-per-task (NEST local_num_threads).
#
# Submit from the repo root:
#     sbatch tc_sleep/slurm/tc_sleep_mn5.sbatch
# Output: out/tc_sleep_mn5.png plus a printed slow-wave (~1 Hz) / spindle
# (~13 Hz) report; the job exits non-zero if either rhythm is absent.
#
# Notes:
#   * Adjust --account/--qos/--partition to your MareNostrum 5 project.
#   * MN5 general-purpose nodes (gpp) have 112 cores; --cpus-per-task up to that.
#   * For a multi-node MPI run, build NEST with MPI, set --nodes/--ntasks and
#     keep --cpus-per-task for threads-per-rank (hybrid MPI+OpenMP). NEST
#     distributes neurons across ranks automatically.

#SBATCH --job-name=tc_sleep
#SBATCH --output=logs/tc_sleep_%j.out
#SBATCH --error=logs/tc_sleep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=01:00:00
###SBATCH --account=<your_bsc_project>
###SBATCH --qos=gp_bscls
###SBATCH --partition=gpp          # general-purpose (Sapphire Rapids)

set -euo pipefail

mkdir -p logs out

# --- environment -----------------------------------------------------------
# On MareNostrum 5, load a Python module and/or activate a venv that has NEST
# 3.x, numpy, scipy and matplotlib. Adjust to your site/NEST install.
# module load python/3.12
# module load nest/3.9            # or: source "$HOME/nest-venv/bin/activate"
# source "$HOME/tc-venv/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

CONFIG="${CONFIG:-tc_sleep/config/network_auditory_mn5.yaml}"
OUTDIR="${OUTDIR:-out}"
TSTOP="${TSTOP:-}"   # empty -> use the value in the YAML (30 s)

echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-1} config=${CONFIG}"

srun python3 tc_sleep/tc_run.py \
    --config "${CONFIG}" \
    --threads "${SLURM_CPUS_PER_TASK:-1}" \
    --outdir "${OUTDIR}" \
    --tag mn5 \
    ${TSTOP:+--tstop "${TSTOP}"}
