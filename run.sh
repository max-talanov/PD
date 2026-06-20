#!/bin/bash -l
#SBATCH --job-name=PD_SWEEP
#SBATCH --output=PD_sweep_%A_%a.slurmout
#SBATCH --error=PD_sweep_%A_%a.slurmerr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-71
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --partition=acc

# SLURM launcher for the PD Phase 5 parameter sweep on BSC MareNostrum 5.
#
# Each array task runs one grid point of pd_phase5_hpc_sweep.py and writes a
# JSON of biomarkers; aggregate them afterwards into a single CSV.
#
# IMPORTANT: keep --array in sync with the grid size. Check it with:
#     python3 pd_phase5_hpc_sweep.py --list
# then set --array=0-(N-1)  (currently 72 points -> 0-71).
#
# Submit:
#     sbatch run.sh
# Aggregate once the array finishes:
#     python3 pd_phase5_hpc_sweep.py --aggregate --outdir results
#
# Adjust --account/--qos for your MareNostrum 5 project. The mean-field
# backend is light (1 core is enough); --cpus-per-task is kept modest so the
# same script also fits the future --backend nest runs.

export LANG=${LANG:-C.UTF-8}
export LC_ALL=${LC_ALL:-C.UTF-8}
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

# --- environment ----------------------------------------------------------
# On MareNostrum 5, load Python and/or activate a venv with numpy (and NEST
# for the spiking backend). Adjust to your site setup.
# module load python/3.12
# source "$HOME/pd-venv/bin/activate"

OUTDIR="${OUTDIR:-results}"
BACKEND="${BACKEND:-meanfield}"
TMAX="${TMAX:-2000}"

mkdir -p "$OUTDIR"

echo "[Slurm] ntasks=$SLURM_NTASKS cpus-per-task=$SLURM_CPUS_PER_TASK array_task=${SLURM_ARRAY_TASK_ID:-NA} backend=$BACKEND"

# environment sanity check
python3 - <<'PY'
import numpy
print("numpy", numpy.__version__)
try:
    import nest
    print("nest", nest.__version__)
except Exception:
    print("nest not available (meanfield backend does not need it)")
PY

srun --cpu-bind=cores --distribution=block:block \
  python3 -u pd_phase5_hpc_sweep.py \
    --run ${SLURM_ARRAY_TASK_ID:-0} \
    --outdir "$OUTDIR" \
    --backend "$BACKEND" \
    --t-max "$TMAX"
