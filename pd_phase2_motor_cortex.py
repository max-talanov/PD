"""
PD Phase 2: Motor cortex integration.

Extends the Phase 1 BGTH network (`pd_phase1_bgth_network.py`) with a
primary motor cortex (M1) population, following bigMC's approach of
chaining a basal-ganglia/thalamus stage into a motor cortex stage
(see bigMC's `tvb_motor_cortex_phase1.py`).

The thalamus (Th) projects excitatory drive to M1. In the PD condition,
the BGTH network's increased GPi->Th inhibition both reduces the mean
thalamocortical drive and lets pathological oscillations (~4-6 Hz,
Parkinsonian tremor band) propagate from STN/GPe through Th into M1.
"""

import numpy as np
import matplotlib.pyplot as plt

from pd_phase1_bgth_network import (
    simulate, firing_rate_hz,
    REGIONS as BGTH_REGIONS,
    PROJECTIONS as BGTH_PROJECTIONS,
    EXTERNAL_DRIVE as BGTH_EXTERNAL_DRIVE,
)

# ----------------------------------------------------------------------
# Extended network: BGTH + M1
# ----------------------------------------------------------------------

REGIONS = BGTH_REGIONS + ["M1"]

PROJECTIONS = BGTH_PROJECTIONS + [
    # Th -> M1 excitation: thalamocortical relay drives motor cortex
    ("Th", "M1", +1, 0.30, 0.15, 5.0),
]

EXTERNAL_DRIVE = dict(BGTH_EXTERNAL_DRIVE)
# Above HH rheobase so M1 fires tonically; thalamocortical (Th -> M1) input
# then modulates this ongoing activity rather than having to ignite it.
EXTERNAL_DRIVE["M1"] = (8.0, 8.0, 1.5)  # baseline cortical drive, unchanged by PD


def run_condition(condition, t_max, dbs_on=False, seed=0):
    return simulate(
        condition=condition, t_max=t_max, dbs_on=dbs_on, seed=seed,
        regions=REGIONS, projections=PROJECTIONS, external_drive=EXTERNAL_DRIVE,
        dbs_target="GPi",
    )


def main():
    t_max = 1000.0  # ms
    print("Running healthy condition...")
    healthy = run_condition("healthy", t_max)
    print("Running PD condition...")
    pd = run_condition("pd", t_max)
    print("Running PD + DBS condition...")
    pd_dbs = run_condition("pd", t_max, dbs_on=True)

    for label, res in [("Healthy", healthy), ("PD", pd), ("PD+DBS", pd_dbs)]:
        print(f"\n-- {label} --")
        for r in REGIONS:
            rate = firing_rate_hz(res["spikes"][r], t_max)
            print(f"  {r}: ~{rate:.1f} Hz (population spike rate)")

    # plot membrane potentials for STN, Th, M1 across conditions
    plot_regions = ["STN", "Th", "M1"]
    fig, axes = plt.subplots(len(plot_regions), 1, figsize=(10, 7), sharex=True)
    for ax, r in zip(axes, plot_regions):
        i = healthy["idx"][r]
        ax.plot(healthy["t"], healthy["v_mean"][i], label="Healthy", alpha=0.8)
        ax.plot(pd["t"], pd["v_mean"][i], label="PD", alpha=0.8)
        ax.plot(pd_dbs["t"], pd_dbs["v_mean"][i], label="PD+DBS", alpha=0.8)
        ax.set_ylabel(f"{r}\nV (mV)")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle("BGTH + motor cortex (M1) prototype")
    fig.tight_layout()
    fig.savefig("pd_phase2_motor_cortex_output.png", dpi=150)
    print("\nSaved plot to pd_phase2_motor_cortex_output.png")

    return {"healthy": healthy, "pd": pd, "pd_dbs": pd_dbs}


if __name__ == "__main__":
    main()
