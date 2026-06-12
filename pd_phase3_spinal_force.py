"""
PD Phase 3: Spinal cord / muscle force and tremor profile.

Extends the Phase 2 BGTH + motor cortex (M1) model
(`pd_phase2_motor_cortex.py`) with a simple spinal cord stage, following
bigMC's progression from cortex to spinal central pattern generator and
limb dynamics (see bigMC's `tvb_phase2_spinal_cpg.py`).

M1 output drives a leaky-integrator alpha-motoneuron pool, whose firing
rate is convolved with a muscle twitch impulse response to produce a
limb force profile. Parkinsonian resting tremor (~4-6 Hz) emerging from
the BGTH loop should appear as oscillations in the PD force trace that
are absent (or much weaker) in the healthy trace, and largely suppressed
under PD+DBS.
"""

import numpy as np
import matplotlib.pyplot as plt

from pd_phase2_motor_cortex import run_condition, REGIONS

TREMOR_BAND = (3.0, 7.0)  # Hz, classic Parkinsonian resting-tremor band


def cortical_drive(res, threshold=-55.0, gain=2.0):
    """Rectified motor-cortex drive signal from M1 mean membrane potential."""
    i = res["idx"]["M1"]
    v = res["v_mean"][i]
    return gain * np.clip(v - threshold, 0.0, None)


def motoneuron_pool_rate(drive, dt, tau_mn=20.0):
    """Leaky-integrator alpha-motoneuron pool firing rate (Hz, arbitrary units)."""
    rate = np.zeros_like(drive)
    for k in range(1, len(drive)):
        rate[k] = rate[k - 1] + dt * (-rate[k - 1] + drive[k - 1]) / tau_mn
    return rate


def twitch_kernel(dt, t_total=200.0, contraction_time=40.0):
    """Single-fiber muscle twitch impulse response (critically damped)."""
    t = np.arange(0, t_total, dt)
    h = (t / contraction_time) * np.exp(1.0 - t / contraction_time)
    return h / h.sum()


def muscle_force(rate, dt):
    h = twitch_kernel(dt)
    force = np.convolve(rate, h)[: len(rate)]
    return force


def dominant_frequency(signal, dt, band=TREMOR_BAND):
    """Return (peak frequency in band, power spectrum, freq axis)."""
    sig = signal - signal.mean()
    n = len(sig)
    freqs = np.fft.rfftfreq(n, d=dt / 1000.0)  # dt in ms -> Hz
    power = np.abs(np.fft.rfft(sig)) ** 2
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    if not band_mask.any() or power[band_mask].sum() == 0:
        return 0.0, power, freqs
    peak_freq = freqs[band_mask][np.argmax(power[band_mask])]
    return peak_freq, power, freqs


def main():
    t_max = 3000.0  # ms; longer run for frequency-domain resolution

    print("Running healthy condition...")
    healthy = run_condition("healthy", t_max)
    print("Running PD condition...")
    pd = run_condition("pd", t_max)
    print("Running PD + DBS condition...")
    pd_dbs = run_condition("pd", t_max, dbs_on=True)

    results = {}
    for label, res in [("Healthy", healthy), ("PD", pd), ("PD+DBS", pd_dbs)]:
        dt = res["dt"]
        drive = cortical_drive(res)
        rate = motoneuron_pool_rate(drive, dt)
        force = muscle_force(rate, dt)
        peak_freq, power, freqs = dominant_frequency(force, dt)
        results[label] = dict(t=res["t"], force=force, power=power, freqs=freqs,
                               peak_freq=peak_freq)
        tremor_power = power[(freqs >= TREMOR_BAND[0]) & (freqs <= TREMOR_BAND[1])].sum()
        total_power = power[1:].sum()  # exclude DC
        print(f"\n-- {label} --")
        print(f"  peak frequency in {TREMOR_BAND} Hz band: {peak_freq:.2f} Hz")
        print(f"  tremor-band power fraction: {tremor_power / total_power:.3f}")

    # plot force traces and power spectra
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    for label in results:
        axes[0].plot(results[label]["t"], results[label]["force"], label=label, alpha=0.8)
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Limb force (a.u.)")
    axes[0].set_title("Limb force profile")
    axes[0].legend()

    for label in results:
        freqs = results[label]["freqs"]
        power = results[label]["power"]
        mask = freqs <= 20.0
        axes[1].plot(freqs[mask], power[mask], label=label, alpha=0.8)
    axes[1].axvspan(*TREMOR_BAND, color="red", alpha=0.1, label="tremor band")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power")
    axes[1].set_title("Force power spectrum (0-20 Hz)")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig("pd_phase3_spinal_force_output.png", dpi=150)
    print("\nSaved plot to pd_phase3_spinal_force_output.png")


if __name__ == "__main__":
    main()
