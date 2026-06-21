"""
PD Phase 6: time-resolved tremor decay at DBS onset.

Phases 4-5 treat DBS as a static condition (on for the whole run), so they
show *how much* tremor DBS removes but not the *transient* -- the decay of
the limb tremor after stimulation is switched on. This phase makes that
visible.

Mechanism (consistent with the rest of the model):
  * The tremor oscillator (Phase 4 Matsuoka) is driven by thalamic
    suppression. Before DBS, thalamus is silenced (PD) -> high tremor drive.
  * At DBS onset, GPi efferent silencing relieves the GPi -> Th inhibition
    and thalamic firing recovers over a time constant ``tau_recover``
    (phenomenological); the tremor drive therefore relaxes from its PD
    level to its PD+DBS level. The Matsuoka oscillation amplitude then
    decays over a few cycles, and the voluntary drive recovers in parallel
    (bradykinesia relief).

Both the PD and the PD+DBS drive levels are read out from the actual
spiking model (Phase 1-2), so the two endpoints of the decay are not
hand-set; only the onset time and recovery time constant are chosen.
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.signal import hilbert
    _HAVE_SCIPY = True
except Exception:                      # pragma: no cover
    _HAVE_SCIPY = False

import pd_phase2_motor_cortex as p2
import pd_phase1_bgth_network as p1
import pd_phase4_spinal_cpg as p4


def _amplitude_envelope(signal, dt, band=p4.TREMOR_BAND):
    """Tremor-band amplitude envelope of a signal."""
    sig = signal - signal.mean()
    if _HAVE_SCIPY:
        # band-pass via FFT mask, then analytic-signal magnitude
        freqs = np.fft.rfftfreq(len(sig), d=dt / 1000.0)
        spec = np.fft.rfft(sig)
        spec[(freqs < band[0]) | (freqs > band[1])] = 0.0
        bandpassed = np.fft.irfft(spec, n=len(sig))
        return np.abs(hilbert(bandpassed))
    # fallback: sliding-window RMS (window ~ 2 tremor cycles)
    win = max(1, int((1000.0 / np.mean(band)) * 2 / dt))
    kernel = np.ones(win) / win
    return np.sqrt(np.convolve(sig ** 2, kernel, mode="same"))


def simulate_dbs_decay(t_max=8000.0, dt=1.0, dbs_onset_ms=3000.0,
                       tau_recover_ms=600.0, seed=0):
    """Single PD trial with GPi-DBS switched on at ``dbs_onset_ms``."""
    n_steps = int(t_max / dt)
    t = np.arange(n_steps) * dt

    # --- endpoint drive levels, read out from the spiking model ----------
    res_pd = p2.run_condition("pd", 2000.0, dbs_on=False)
    res_dbs = p2.run_condition("pd", 2000.0, dbs_on=True)   # GPi-DBS (default target)

    s_pd = p4.tremor_drive(res_pd, 2000.0)
    s_dbs = p4.tremor_drive(res_dbs, 2000.0)
    v_pd = p4.descending_command(res_pd, 2000.0)
    v_dbs = p4.descending_command(res_dbs, 2000.0)

    # --- time-varying drives: step at onset, exponential relaxation ------
    after = np.clip(t - dbs_onset_ms, 0.0, None)
    relax = np.where(t >= dbs_onset_ms, np.exp(-after / tau_recover_ms), 1.0)
    s_trem_t = s_dbs + (s_pd - s_dbs) * relax       # tremor drive decays
    v_cmd_t = v_dbs + (v_pd - v_dbs) * relax        # voluntary drive recovers

    # --- rhythm generators with time-varying drive (Phase 4 Matsuoka) ----
    trem_a, trem_b = p4.matsuoka(s_trem_t, dt, tr=6.0, ta=98.0)     # ~5 Hz
    tremor = trem_a - trem_b
    cpg_flex, cpg_ext = p4.matsuoka(v_cmd_t, dt, tr=10.0, ta=650.0)  # ~1.5 Hz

    flex_cmd = cpg_flex + 0.5 * tremor
    ext_cmd = cpg_ext - 0.5 * tremor
    net = p4.motoneuron_force(flex_cmd, dt) - p4.motoneuron_force(ext_cmd, dt)

    env = _amplitude_envelope(net, dt)
    return {
        "t": t, "net": net, "env": env,
        "s_trem_t": s_trem_t, "v_cmd_t": v_cmd_t,
        "dbs_onset_ms": dbs_onset_ms, "tau_recover_ms": tau_recover_ms,
        "s_pd": s_pd, "s_dbs": s_dbs,
    }


def _decay_time(t, env, dbs_onset_ms):
    """Time after onset for the tremor envelope to fall to 1/e of its drop."""
    pre = env[t < dbs_onset_ms]
    post = env[t >= dbs_onset_ms]
    tpost = t[t >= dbs_onset_ms]
    if len(pre) == 0 or len(post) == 0:
        return float("nan")
    a0 = pre.mean()
    a_inf = post[-int(len(post) * 0.2):].mean()  # final plateau
    target = a_inf + (a0 - a_inf) / np.e
    below = np.where(post <= target)[0]
    return (tpost[below[0]] - dbs_onset_ms) if len(below) else float("nan")


def main():
    r = simulate_dbs_decay()
    t = r["t"]
    tau63 = _decay_time(t, r["env"], r["dbs_onset_ms"])
    pre = r["env"][t < r["dbs_onset_ms"]].mean()
    post = r["env"][t >= r["dbs_onset_ms"]][-200:].mean()
    print(f"DBS onset at {r['dbs_onset_ms']:.0f} ms "
          f"(thalamic-recovery tau = {r['tau_recover_ms']:.0f} ms)")
    print(f"tremor drive: PD {r['s_pd']:.2f} -> PD+DBS {r['s_dbs']:.2f}")
    print(f"tremor envelope: pre-DBS {pre:.3f} -> post-DBS plateau {post:.3f} "
          f"({100 * (1 - post / pre):.0f}% reduction)")
    print(f"envelope 1/e decay time after onset: ~{tau63:.0f} ms")

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(t, r["net"], color="tab:purple", lw=0.8, label="limb force")
    axes[0].plot(t, r["env"], color="black", lw=1.6, label="tremor envelope")
    axes[0].plot(t, -r["env"], color="black", lw=1.6)
    axes[0].axvline(r["dbs_onset_ms"], color="tab:green", ls="--", lw=2,
                    label="GPi-DBS on")
    axes[0].set_ylabel("net limb force (a.u.)")
    axes[0].set_title("Tremor decay at DBS onset")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(t, r["s_trem_t"], color="tab:red", label="tremor drive")
    axes[1].plot(t, r["v_cmd_t"], color="tab:blue", label="voluntary drive")
    axes[1].axvline(r["dbs_onset_ms"], color="tab:green", ls="--", lw=2)
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("drive (a.u.)")
    axes[1].set_title("DBS relieves thalamic suppression: tremor drive falls, "
                      "voluntary drive recovers")
    axes[1].legend(loc="center right", fontsize=8)

    fig.tight_layout()
    fig.savefig("pd_phase6_dbs_tremor_decay_output.png", dpi=150)
    print("\nSaved plot to pd_phase6_dbs_tremor_decay_output.png")


if __name__ == "__main__":
    main()
