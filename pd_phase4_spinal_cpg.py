"""
PD Phase 4: Spinal CPG + limb force / tremor.

Goal: actually *see* Parkinsonian rest tremor in a limb force profile.

Why the earlier phases didn't show tremor
------------------------------------------
The mean-field Hodgkin-Huxley nuclei (phases 1-2) fire tonically but
asynchronously, so their population *mean membrane potential* carries
essentially no coherent low-frequency oscillation (tremor-band power ~0).
Forcing a 10-neuron spiking pool to self-organise into a clean limit cycle
is a hard, brittle tuning problem. Following the multiscale co-simulation
philosophy of the source article (microscale neurons + macroscale
dynamics), this phase places the *rhythm generators* at the population /
rate level, where they are robust and controllable:

  * a spinal central pattern generator (CPG) -- a Matsuoka flexor/extensor
    half-center oscillator, the same reciprocal-inhibition-with-fatigue
    mechanism used by tinyCPG (https://github.com/max-talanov/tinyCPG) --
    produces the rhythmic *voluntary* limb drive (repetitive "steps");
  * a second Matsuoka oscillator tuned to the ~4-6 Hz tremor band acts as
    the pathological basal-ganglia tremor source.

Coupling to the spiking model (phases 1-2)
------------------------------------------
Both rhythm generators are *driven by* the BGTH + M1 spiking network:

  * voluntary CPG drive  <- thalamocortical output (thalamus firing rate).
    PD collapses thalamic firing -> weaker, slower voluntary force
    (bradykinesia/hypokinesia).
  * tremor drive         <- STN hyperactivity above a healthy reference.
    Parkinsonian STN over-activity feeds the tremor oscillator. DBS is
    modelled as functional suppression of the pathological STN output
    reaching the tremor circuit (the "informational lesion" view of DBS),
    so it reduces the tremor drive.

The tremor oscillation is injected *common-mode* into both motoneuron
pools (it shakes the limb) while the CPG drives them in *antagonist*
fashion (it moves the limb). Each motoneuron pool is low-pass filtered and
convolved with a muscle twitch to give flexor / extensor force; the net
joint force is their difference.
"""

import numpy as np
import matplotlib.pyplot as plt

from pd_phase2_motor_cortex import run_condition, REGIONS
from pd_phase1_bgth_network import firing_rate_hz
from pd_phase3_spinal_force import twitch_kernel, TREMOR_BAND

# Healthy reference thalamic rate (Hz); PD *suppression* below this drives
# both bradykinesia (weak voluntary drive) and tremor (rebound bursting).
# DBS reduces tremor *emergently*: by relieving GPi over-inhibition it
# restores thalamic firing in the spiking model, which lowers this
# suppression term -- no separate DBS factor is needed here.
TH_HEALTHY_REF = 215.0


# ----------------------------------------------------------------------
# Matsuoka half-center oscillator (reciprocal inhibition + fatigue)
# ----------------------------------------------------------------------

def matsuoka(drive, dt, tr, ta, a=2.5, b=2.5, n_steps=None, seed=0):
    """Two mutually-inhibiting fatiguing neurons -> stable limit cycle.

    drive : tonic excitation (scalar or array of length n_steps). Oscillation
            amplitude grows with drive; below a threshold the system is quiet.
    tr, ta : rise and adaptation time constants (ms) -> set the frequency.
    Returns (y0, y1): the two half-center outputs (>=0).
    """
    if np.isscalar(drive):
        assert n_steps is not None
        drive = np.full(n_steps, float(drive))
    n = len(drive)
    x = np.zeros((2, n))
    v = np.zeros((2, n))
    y = np.zeros((2, n))
    x[:, 0] = [0.1, -0.1]  # asymmetric start to break the symmetry
    for k in range(1, n):
        for i in range(2):
            j = 1 - i
            dx = (-x[i, k - 1] - a * y[j, k - 1] - b * v[i, k - 1] + drive[k - 1]) / tr
            dv = (-v[i, k - 1] + y[i, k - 1]) / ta
            x[i, k] = x[i, k - 1] + dt * dx
            v[i, k] = v[i, k - 1] + dt * dv
            y[i, k] = max(0.0, x[i, k])
    return y[0], y[1]


# ----------------------------------------------------------------------
# Drives derived from the BGTH + M1 spiking model
# ----------------------------------------------------------------------

def descending_command(res, t_max):
    """Voluntary drive to the spinal CPG, from thalamocortical output."""
    th_rate = firing_rate_hz(res["spikes"]["Th"], t_max)
    # normalise: healthy Th ~190 Hz -> ~1.5 drive; collapsed PD Th -> ~0.
    return np.clip(th_rate / 130.0, 0.0, 2.5)


def tremor_drive(res, t_max, th_ref=TH_HEALTHY_REF):
    """Pathological tremor drive, from thalamic suppression below healthy ref.

    PD collapses thalamic firing (GPi over-inhibition); the resulting
    thalamic disinhibition / rebound bursting feeds the tremor oscillator.
    """
    th_rate = firing_rate_hz(res["spikes"]["Th"], t_max)
    suppression = max(0.0, th_ref - th_rate)
    drive = suppression / 120.0  # scale into the Matsuoka oscillation range
    return drive


# ----------------------------------------------------------------------
# Spinal cord -> muscle -> limb force
# ----------------------------------------------------------------------

def motoneuron_force(command, dt, tau_mn=20.0):
    """Leaky motoneuron pool + muscle twitch -> force for one muscle."""
    rate = np.zeros_like(command)
    for k in range(1, len(command)):
        rate[k] = rate[k - 1] + dt * (-rate[k - 1] + command[k - 1]) / tau_mn
    h = twitch_kernel(dt)
    return np.convolve(rate, h)[: len(rate)]


def limb_force(res, t_max, dt, th_ref=TH_HEALTHY_REF,
               cpg_tr=10.0, cpg_ta=650.0,      # ~1.5 Hz voluntary rhythm
               trem_tr=6.0, trem_ta=98.0):     # ~5 Hz tremor
    """Compute net joint force time series for one condition.

    The DBS effect enters through ``res`` (the spiking model already encodes
    DBS): GPi-DBS restores thalamic firing, which lowers both the tremor
    drive and the voluntary-drive deficit.
    """
    n_steps = int(t_max / dt)

    # voluntary rhythm: CPG drives flexor/extensor in antagonist fashion
    v_cmd = descending_command(res, t_max)
    cpg_flex, cpg_ext = matsuoka(v_cmd, dt, cpg_tr, cpg_ta, n_steps=n_steps)

    # pathological tremor: common-mode shake added to both pools
    s_trem = tremor_drive(res, t_max, th_ref=th_ref)
    trem_a, trem_b = matsuoka(s_trem, dt, trem_tr, trem_ta, n_steps=n_steps)
    tremor = trem_a - trem_b  # zero-mean oscillation

    flex_cmd = cpg_flex + 0.5 * tremor
    ext_cmd = cpg_ext - 0.5 * tremor

    f_flex = motoneuron_force(flex_cmd, dt)
    f_ext = motoneuron_force(ext_cmd, dt)
    net = f_flex - f_ext  # net joint force / torque
    return {"t": np.arange(n_steps) * dt, "net": net,
            "v_cmd": v_cmd, "s_trem": s_trem}


def spectrum(signal, dt, band=TREMOR_BAND):
    sig = signal - signal.mean()
    freqs = np.fft.rfftfreq(len(sig), d=dt / 1000.0)
    power = np.abs(np.fft.rfft(sig)) ** 2
    mask = (freqs >= band[0]) & (freqs <= band[1])
    peak = freqs[mask][np.argmax(power[mask])] if mask.any() else 0.0
    frac = power[mask].sum() / power[1:].sum() if power[1:].sum() else 0.0
    return freqs, power, peak, frac


def main():
    t_max = 4000.0  # ms, several voluntary cycles for frequency resolution
    dt = 1.0        # ms; rate-level integration step

    print("Running BGTH + M1 spiking model per condition...")
    conditions = [
        ("Healthy", "healthy", False),
        ("PD", "pd", False),
        ("PD+DBS", "pd", True),
    ]

    # spiking runs first; use the healthy STN rate as the tremor reference so
    # the healthy condition has (by construction) ~no pathological tremor drive
    spiking = {label: run_condition(cond, t_max, dbs_on=dbs)
               for label, cond, dbs in conditions}
    th_ref = firing_rate_hz(spiking["Healthy"]["spikes"]["Th"], t_max)

    results = {}
    for label, cond, dbs in conditions:
        res = spiking[label]
        lf = limb_force(res, t_max, dt, th_ref=th_ref)
        freqs, power, peak, frac = spectrum(lf["net"], dt)
        # absolute tremor-band amplitude (RMS of band-passed force)
        mask = (freqs >= TREMOR_BAND[0]) & (freqs <= TREMOR_BAND[1])
        band_rms = np.sqrt(power[mask].sum()) / len(lf["net"])
        results[label] = dict(lf=lf, freqs=freqs, power=power, peak=peak,
                               frac=frac, band_rms=band_rms)
        print(f"\n-- {label} --")
        print(f"  voluntary CPG drive: {lf['v_cmd']:.2f}   tremor drive: {lf['s_trem']:.3f}")
        print(f"  tremor-band peak: {peak:.2f} Hz")
        print(f"  tremor-band power fraction: {frac:.3f}")
        print(f"  tremor amplitude (band RMS, absolute): {band_rms:.4f}")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    for label in results:
        lf = results[label]["lf"]
        axes[0].plot(lf["t"], lf["net"], label=label, alpha=0.85)
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Net limb force (a.u.)")
    axes[0].set_title("Limb force profile: voluntary rhythm + Parkinsonian tremor")
    axes[0].legend()

    for label in results:
        freqs = results[label]["freqs"]
        power = results[label]["power"]
        mask = freqs <= 12.0
        axes[1].plot(freqs[mask], power[mask], label=label, alpha=0.85)
    axes[1].axvspan(*TREMOR_BAND, color="red", alpha=0.1, label="tremor band")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power")
    axes[1].set_title("Limb force power spectrum (0-12 Hz)")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig("pd_phase4_spinal_cpg_output.png", dpi=150)
    print("\nSaved plot to pd_phase4_spinal_cpg_output.png")


if __name__ == "__main__":
    main()
