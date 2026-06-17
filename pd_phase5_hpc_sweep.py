"""
PD Phase 5: HPC parameter sweep driver (MareNostrum 5 / BSC ready).

This phase does not add new biology; it wraps the Phase 1-4 model in an
*embarrassingly parallel* parameter sweep so it can run as a SLURM job
array on a supercomputer (e.g. BSC MareNostrum 5) and, later, swap the
small mean-field backend for a large-scale spiking backend (NEST) behind
the same interface.

Two responsibilities:

  1. Map a flat integer index (e.g. $SLURM_ARRAY_TASK_ID) to one point of a
     parameter grid, run the model, and write that point's biomarkers to a
     small JSON file. Thousands of these run independently across nodes.
  2. Aggregate all per-task JSON files into a single CSV for downstream
     analysis / fitting (parameter inference against measured biomarkers).

Usage
-----
Local smoke test (runs a few grid points serially)::

    python pd_phase5_hpc_sweep.py --list                 # show grid size
    python pd_phase5_hpc_sweep.py --run 0 --outdir out   # run one point
    python pd_phase5_hpc_sweep.py --smoke --outdir out   # run first 4 points
    python pd_phase5_hpc_sweep.py --aggregate --outdir out

On the cluster, a SLURM array launches one ``--run $SLURM_ARRAY_TASK_ID``
per task (see slurm/pd_phase5_sweep.sbatch).

Backends
--------
``--backend meanfield`` (default) uses the Phase 1-2 Hodgkin-Huxley
mean-field network in this repo, so the whole pipeline runs on a laptop.
``--backend nest`` is a documented stub for the large-scale spiking model
to be implemented on the cluster (see ``run_nest_backend``).
"""

import argparse
import itertools
import json
import os
import glob

import numpy as np

import pd_phase2_motor_cortex as p2
import pd_phase1_bgth_network as p1
import pd_phase4_spinal_cpg as p4


# ----------------------------------------------------------------------
# Parameter grid
# ----------------------------------------------------------------------
# Each axis is a biophysically meaningful knob we currently hand-tune. On
# HPC we sweep them (and, later, replace this grid with simulation-based
# inference that fits these to measured per-subject biomarkers).

GRID = {
    "condition": ["healthy", "pd"],
    "gpi_th_pd_gain": [0.55, 0.70, 0.85, 1.00],   # GPi -> Th inhibition in PD
    "gpe_stn_delay_ms": [10.0, 25.0, 40.0],       # STN <-> GPe loop delay
    "dbs_target": ["none", "GPi", "STN"],         # DBS target ("none" = no DBS)
    "dbs_suppression": [0.5, 0.8],                # efferent silencing fraction
}

GRID_KEYS = list(GRID.keys())


def grid_points():
    """All parameter combinations as a list of dicts (stable order)."""
    combos = itertools.product(*(GRID[k] for k in GRID_KEYS))
    points = [dict(zip(GRID_KEYS, c)) for c in combos]
    # prune redundant combinations: dbs params are irrelevant when no DBS,
    # and DBS is only meaningful in the PD condition.
    pruned = []
    seen = set()
    for p in points:
        q = dict(p)
        if q["dbs_target"] == "none" or q["condition"] == "healthy":
            q["dbs_target"] = "none"
            q["dbs_suppression"] = 0.0
        key = tuple(q[k] for k in GRID_KEYS)
        if key not in seen:
            seen.add(key)
            pruned.append(q)
    return pruned


# ----------------------------------------------------------------------
# Mean-field backend (this repo) + biomarker extraction
# ----------------------------------------------------------------------

def build_projections(gpi_th_pd_gain, gpe_stn_delay_ms):
    """Copy the Phase 2 projection list with two swept values overridden."""
    proj = []
    for p in p2.PROJECTIONS:
        p = list(p)
        if (p[0], p[1]) == ("GPi", "Th"):
            p[4] = gpi_th_pd_gain           # pd_g
        if (p[0], p[1]) == ("GPe", "STN"):
            p[6] = gpe_stn_delay_ms         # delay_ms
        proj.append(tuple(p))
    return proj


def run_meanfield_backend(params, t_max=2000.0, dt=1.0, seed=0):
    """Run one parameter point and return a biomarker dict."""
    condition = params["condition"]
    dbs_on = params["dbs_target"] != "none"
    dbs_target = params["dbs_target"] if dbs_on else "GPi"

    projections = build_projections(params["gpi_th_pd_gain"],
                                    params["gpe_stn_delay_ms"])

    # DBS efferent-silencing strength is a module-level constant; set it for
    # this run (each SLURM task is its own process, so this is safe).
    p1.DBS_EFFERENT_SUPPRESSION = params["dbs_suppression"]

    res = p1.simulate(
        condition=condition, t_max=t_max, seed=seed,
        regions=p2.REGIONS, projections=projections,
        external_drive=p2.EXTERNAL_DRIVE,
        dbs_on=dbs_on, dbs_target=dbs_target,
    )

    rates = {f"rate_{r}": float(p1.firing_rate_hz(res["spikes"][r], t_max))
             for r in p2.REGIONS}

    # limb force / tremor read-out (Phase 4)
    lf = p4.limb_force(res, t_max, dt)
    freqs, power, peak, frac = p4.spectrum(lf["net"], dt)
    mask = (freqs >= p4.TREMOR_BAND[0]) & (freqs <= p4.TREMOR_BAND[1])
    band_rms = float(np.sqrt(power[mask].sum()) / len(lf["net"]))

    return {
        **rates,
        "voluntary_drive": float(lf["v_cmd"]),
        "tremor_drive": float(lf["s_trem"]),
        "tremor_peak_hz": float(peak),
        "tremor_band_fraction": float(frac),
        "tremor_band_rms": band_rms,
    }


def run_nest_backend(params, **kwargs):
    """Large-scale spiking backend (to be implemented on the cluster).

    Intended mapping from the mean-field model:
      * each nucleus -> a population of 1e3-1e5 conductance-based spiking
        neurons (NEST ``aeif_cond_*`` or Hodgkin-Huxley models);
      * PROJECTIONS -> sparse probabilistic connections with the same
        sign/gain/delay, scaled by population size;
      * EXTERNAL_DRIVE -> Poisson generators;
      * DBS -> high-frequency stimulation + efferent synaptic depression on
        the target population;
      * biomarkers -> identical keys (population rates, tremor read-out) so
        the aggregation/inference code is backend-agnostic.
    Run with NEST's MPI+OpenMP across the MareNostrum 5 GPP partition (or a
    GPU spiking simulator on the ACC/H100 partition).
    """
    raise NotImplementedError(
        "NEST backend is a cluster-side stub; see docstring for the "
        "mean-field -> spiking mapping. Use --backend meanfield locally."
    )


BACKENDS = {"meanfield": run_meanfield_backend, "nest": run_nest_backend}


# ----------------------------------------------------------------------
# Task running + aggregation
# ----------------------------------------------------------------------

def run_index(index, outdir, backend="meanfield", **kwargs):
    points = grid_points()
    if not 0 <= index < len(points):
        raise IndexError(f"index {index} out of range 0..{len(points) - 1}")
    params = points[index]
    metrics = BACKENDS[backend](params, **kwargs)
    record = {"index": index, "backend": backend, "params": params,
              "metrics": metrics}
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"task_{index:06d}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return path, record


def aggregate(outdir, csv_path=None):
    """Merge all per-task JSON files into one CSV."""
    import csv
    files = sorted(glob.glob(os.path.join(outdir, "task_*.json")))
    if not files:
        raise FileNotFoundError(f"no task_*.json files in {outdir}")
    rows = []
    for fp in files:
        with open(fp) as f:
            rec = json.load(f)
        row = {"index": rec["index"], "backend": rec["backend"]}
        row.update(rec["params"])
        row.update(rec["metrics"])
        rows.append(row)
    csv_path = csv_path or os.path.join(outdir, "sweep_results.csv")
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return csv_path, len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print grid size and exit")
    ap.add_argument("--run", type=int, metavar="INDEX",
                    help="run a single grid point (e.g. $SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--smoke", action="store_true",
                    help="run the first few grid points serially (local test)")
    ap.add_argument("--aggregate", action="store_true",
                    help="merge per-task JSON files into a CSV")
    ap.add_argument("--outdir", default="sweep_out", help="output directory")
    ap.add_argument("--backend", default="meanfield", choices=list(BACKENDS))
    ap.add_argument("--t-max", type=float, default=2000.0)
    args = ap.parse_args()

    points = grid_points()
    if args.list:
        print(f"grid axes: {GRID_KEYS}")
        print(f"total grid points (after pruning): {len(points)}")
        return

    if args.run is not None:
        path, rec = run_index(args.run, args.outdir, backend=args.backend,
                              t_max=args.t_max)
        print(f"[{args.run}] {rec['params']}")
        print(f"  -> {rec['metrics']}")
        print(f"  wrote {path}")
        return

    if args.smoke:
        n = min(4, len(points))
        print(f"smoke test: running first {n} of {len(points)} points...")
        for i in range(n):
            _, rec = run_index(i, args.outdir, backend=args.backend,
                              t_max=args.t_max)
            print(f"  [{i}] {rec['params']['condition']} "
                  f"dbs={rec['params']['dbs_target']} "
                  f"Th={rec['metrics']['rate_Th']:.0f}Hz "
                  f"tremor_rms={rec['metrics']['tremor_band_rms']:.4f}")
        path, n = aggregate(args.outdir)
        print(f"aggregated {n} rows -> {path}")
        return

    if args.aggregate:
        path, n = aggregate(args.outdir)
        print(f"aggregated {n} rows -> {path}")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
