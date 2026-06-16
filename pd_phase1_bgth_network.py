"""
PD Phase 1: Basal Ganglia-Thalamus (BGTH) network prototype.

Inspired by:
  - bigMC (https://github.com/max-talanov/bigMC): phase-based simulation
    prototypes of neural circuits with healthy vs. pathological states
    and stimulation experiments.
  - Multiscale BGTH model of Parkinson's disease and DBS
    (https://pmc.ncbi.nlm.nih.gov/articles/PMC12458589/): four coupled
    nuclei (STN, GPe, GPi, Thalamus) of Hodgkin-Huxley-type neurons with
    physiological excitatory/inhibitory pathways, dopamine-dependent
    coupling strengths, stochastic drive, and open-loop DBS.

This is a first, simplified prototype: each nucleus is a small population
of conductance-based Hodgkin-Huxley neurons coupled via mean-field
exponential synapses. Two parameter regimes (healthy / PD) are provided,
plus an optional high-frequency DBS pulse train applied to STN.
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Hodgkin-Huxley kinetics (standard squid-axon style, in mV / ms / uF/cm^2)
# ----------------------------------------------------------------------

C_M = 1.0      # membrane capacitance (uF/cm^2)
G_NA, E_NA = 120.0, 50.0
G_K, E_K = 36.0, -77.0
G_L, E_L = 0.3, -54.4


def alpha_m(v):
    return 0.1 * (v + 40.0) / (1.0 - np.exp(-(v + 40.0) / 10.0))


def beta_m(v):
    return 4.0 * np.exp(-(v + 65.0) / 18.0)


def alpha_h(v):
    return 0.07 * np.exp(-(v + 65.0) / 20.0)


def beta_h(v):
    return 1.0 / (1.0 + np.exp(-(v + 35.0) / 10.0))


def alpha_n(v):
    return 0.01 * (v + 55.0) / (1.0 - np.exp(-(v + 55.0) / 10.0))


def beta_n(v):
    return 0.125 * np.exp(-(v + 65.0) / 80.0)


def hh_currents(v, m, h, n):
    i_na = G_NA * m ** 3 * h * (v - E_NA)
    i_k = G_K * n ** 4 * (v - E_K)
    i_l = G_L * (v - E_L)
    return i_na + i_k + i_l


# ----------------------------------------------------------------------
# Network definition: STN, GPe, GPi, Thalamus (Th)
# ----------------------------------------------------------------------

REGIONS = ["STN", "GPe", "GPi", "Th"]

# Synaptic projections: (source, target, sign, healthy_g, pd_g, tau_ms, delay_ms)
# sign: +1 excitatory (E_syn = 0 mV), -1 inhibitory (E_syn = -80 mV)
# delay_ms: axonal conduction + synaptic delay (the reciprocal STN <-> GPe
# loop is a delayed feedback loop). In PD the indirect-pathway gains are
# increased (g_pd): STN/GPi drive grows and GPi over-inhibits the thalamus,
# collapsing thalamocortical output -- the core basal-ganglia output
# abnormality that Phase 4 reads out to drive limb bradykinesia and tremor.
PROJECTIONS = [
    ("STN", "GPe", +1, 0.15, 0.30, 5.0, 6.0),    # STN -> GPe excitation
    ("GPe", "STN", -1, 0.20, 0.55, 10.0, 10.0),  # GPe -> STN inhibition (delayed, stronger in PD)
    ("GPe", "GPi", -1, 0.15, 0.15, 10.0, 6.0),   # GPe -> GPi inhibition
    ("STN", "GPi", +1, 0.15, 0.35, 5.0, 6.0),    # STN -> GPi excitation (strengthens in PD)
    ("GPi", "Th", -1, 0.25, 0.55, 10.0, 5.0),    # GPi -> Th inhibition (strengthens in PD)
]

# External drive (cortex/striatum input) per region: (healthy_mean, pd_mean, std)
# Values are kept above the Hodgkin-Huxley rheobase (~6.2 uA/cm^2) so each
# nucleus fires *tonically* (repetitive spike trains) rather than emitting a
# single onset transient and then resting. The std term adds per-neuron noise
# so the populations desynchronize into sustained, ongoing activity.
EXTERNAL_DRIVE = {
    "STN": (10.0, 14.0, 1.5),  # PD: increased excitatory drive to STN (loss of striatal D2 inhibition)
    "GPe": (12.0, 8.0, 1.5),   # PD: reduced drive to GPe (less striatal inhibition relief)
    "GPi": (10.0, 13.0, 1.5),
    "Th": (9.0, 6.0, 1.5),     # PD: reduced thalamocortical drive
}

E_EXC, E_INH = 0.0, -80.0


# ----------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------

def simulate(condition="healthy", n_per_region=10, t_max=1000.0, dt=0.01,
              dbs_on=False, dbs_freq=130.0, dbs_amp=3.0, seed=0,
              regions=None, projections=None, external_drive=None,
              dbs_target="STN"):
    """Simulate a Hodgkin-Huxley mean-field network.

    condition : "healthy" or "pd"
    n_per_region : number of HH neurons per nucleus (mean-field coupled)
    t_max, dt : simulation time and step (ms)
    dbs_on : apply high-frequency square-pulse DBS current to dbs_target
    dbs_freq, dbs_amp : DBS pulse frequency (Hz) and amplitude (uA/cm^2)
    regions, projections, external_drive : override the default BGTH
        network so callers (e.g. phase2) can extend it with extra nuclei.
    """
    regions = regions if regions is not None else REGIONS
    projections = projections if projections is not None else PROJECTIONS
    external_drive = external_drive if external_drive is not None else EXTERNAL_DRIVE

    rng = np.random.default_rng(seed)
    n_steps = int(t_max / dt)
    n_reg = len(regions)
    idx = {r: i for i, r in enumerate(regions)}

    # state arrays: shape (n_regions, n_per_region)
    v = -65.0 + 5.0 * rng.standard_normal((n_reg, n_per_region))
    m = alpha_m(v) / (alpha_m(v) + beta_m(v))
    h = alpha_h(v) / (alpha_h(v) + beta_h(v))
    n = alpha_n(v) / (alpha_n(v) + beta_n(v))

    # normalise projections to 7-tuples (default zero delay if unspecified)
    projections = [p if len(p) == 7 else (*p, 0.0) for p in projections]

    # synaptic gating variable per projection (mean-field, scalar)
    s_syn = {p[:2]: 0.0 for p in projections}

    # per-projection conduction delay in integration steps
    delay_steps = {p[:2]: int(round(p[6] / dt)) for p in projections}

    # history of each region's firing fraction, for delayed synaptic readout
    fire_hist = np.zeros((n_reg, n_steps))

    # outputs
    t = np.arange(n_steps) * dt
    v_mean = np.zeros((n_reg, n_steps))
    spikes = {r: [] for r in regions}

    above_thresh_prev = np.zeros((n_reg, n_per_region), dtype=bool)

    # DBS pulse train (square pulses on dbs_target)
    dbs_period = 1000.0 / dbs_freq if dbs_on else None
    pulse_width = 0.3  # ms

    for step in range(n_steps):
        time = step * dt

        # --- external + noise drive per region ---
        i_ext = np.zeros((n_reg, n_per_region))
        for r in regions:
            mean, pd_mean, std = external_drive[r]
            base = pd_mean if condition == "pd" else mean
            i_ext[idx[r]] = base + std * rng.standard_normal(n_per_region)

        # --- DBS current to target nucleus ---
        if dbs_on:
            phase = time % dbs_period
            if phase < pulse_width:
                i_ext[idx[dbs_target]] += dbs_amp

        # current firing fraction per region (for the delay history buffer)
        fire_hist[:, step] = above_thresh_prev.mean(axis=1)

        # --- synaptic currents from projections (with conduction delay) ---
        i_syn = np.zeros((n_reg, n_per_region))
        for (src, tgt, sign, g_h, g_pd, tau, _delay) in projections:
            g = g_pd if condition == "pd" else g_h
            e_syn = E_EXC if sign > 0 else E_INH
            s = s_syn[(src, tgt)]
            i_syn[idx[tgt]] += g * s * (v[idx[tgt]] - e_syn)
            # synapse gating decays exponentially, driven by the *delayed*
            # source firing rate (axonal conduction + synaptic delay)
            d = delay_steps[(src, tgt)]
            firing_frac = fire_hist[idx[src], step - d] if step >= d else 0.0
            s_syn[(src, tgt)] = s + dt * (firing_frac * 5.0 * (1 - s) - s / tau)

        # --- HH dynamics (Euler integration) ---
        i_total = i_ext - i_syn - hh_currents(v, m, h, n)
        dv = i_total / C_M
        dm = alpha_m(v) * (1 - m) - beta_m(v) * m
        dh = alpha_h(v) * (1 - h) - beta_h(v) * h
        dn = alpha_n(v) * (1 - n) - beta_n(v) * n

        v = v + dt * dv
        m = np.clip(m + dt * dm, 0.0, 1.0)
        h = np.clip(h + dt * dh, 0.0, 1.0)
        n = np.clip(n + dt * dn, 0.0, 1.0)

        # spike detection (rising edge through 0 mV)
        above_thresh = v > 0.0
        rising = above_thresh & ~above_thresh_prev
        for r in regions:
            if rising[idx[r]].any():
                spikes[r].append(time)
        above_thresh_prev = above_thresh

        for r in regions:
            v_mean[idx[r], step] = v[idx[r]].mean()

    return {"t": t, "v_mean": v_mean, "spikes": spikes, "idx": idx,
            "regions": regions, "dt": dt, "n_per_region": n_per_region}


def firing_rate_hz(spike_times, t_max):
    """Approximate population firing rate (Hz) from recorded spike events."""
    return 1000.0 * len(spike_times) / t_max


def main():
    t_max = 1000.0  # ms
    print("Running healthy condition...")
    healthy = simulate(condition="healthy", t_max=t_max)
    print("Running PD condition...")
    pd = simulate(condition="pd", t_max=t_max)
    print("Running PD + DBS condition...")
    pd_dbs = simulate(condition="pd", t_max=t_max, dbs_on=True)

    for label, res in [("Healthy", healthy), ("PD", pd), ("PD+DBS", pd_dbs)]:
        print(f"\n-- {label} --")
        for r in REGIONS:
            rate = firing_rate_hz(res["spikes"][r], t_max)
            print(f"  {r}: ~{rate:.1f} Hz (population spike rate)")

    # plot membrane potentials of STN and Th across conditions
    fig, axes = plt.subplots(len(REGIONS), 1, figsize=(10, 8), sharex=True)
    for ax, r in zip(axes, REGIONS):
        i = healthy["idx"][r]
        ax.plot(healthy["t"], healthy["v_mean"][i], label="Healthy", alpha=0.8)
        ax.plot(pd["t"], pd["v_mean"][i], label="PD", alpha=0.8)
        ax.plot(pd_dbs["t"], pd_dbs["v_mean"][i], label="PD+DBS", alpha=0.8)
        ax.set_ylabel(f"{r}\nV (mV)")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle("BGTH network prototype: mean membrane potential per nucleus")
    fig.tight_layout()
    fig.savefig("pd_phase1_bgth_output.png", dpi=150)
    print("\nSaved plot to pd_phase1_bgth_output.png")


if __name__ == "__main__":
    main()
