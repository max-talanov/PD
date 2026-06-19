"""
Driver for the auditory thalamo-cortical sleep model.

Builds the closed-loop column (tc_network.AuditoryThalamoCorticalSleep), runs
it in NEST, and verifies the two required rhythms:

  * slow-wave  ~1 Hz   -- a peak near 1 Hz in the PSD of the cortical
                          population-rate (LFP-proxy) signal.
  * spindles   ~13 Hz  -- power in the 11-15 Hz band that waxes/wanes with the
                          slow oscillation, shown in a spectrogram.

Run locally (sane test, seconds)::

    python tc_sleep/tc_run.py --config tc_sleep/config/network_auditory_local.yaml --outdir out

On MareNostrum 5 the same entry-point is launched with the MN5 config and
multiple threads (see tc_sleep/slurm/tc_sleep_mn5.sbatch).

Exit status is non-zero if either rhythm is missing, so the run self-validates.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# allow both `python tc_sleep/tc_run.py` and `python -m tc_sleep.tc_run`
try:
    from tc_sleep.tc_network import (
        AuditoryThalamoCorticalSleep, NetworkConfig, SimulationConfig,
        SynapseParams, SleepParams,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tc_network import (  # type: ignore
        AuditoryThalamoCorticalSleep, NetworkConfig, SimulationConfig,
        SynapseParams, SleepParams,
    )


# ---------------------------------------------------------------------------
#  Signal construction + analysis
# ---------------------------------------------------------------------------

def population_rate(spike_times, tstop, bin_ms=2.0, smooth_ms=10.0):
    """Binned, lightly smoothed population firing-rate signal (an LFP-proxy).

    Returns (t_centres_ms, rate) sampled at ``bin_ms``.
    """
    n_bins = max(2, int(np.ceil(tstop / bin_ms)))
    edges = np.arange(n_bins + 1) * bin_ms
    counts, _ = np.histogram(spike_times, bins=edges)
    rate = counts.astype(float) / (bin_ms / 1000.0)  # spikes/s (unnormalised)
    # Gaussian smoothing
    if smooth_ms > 0:
        sigma = smooth_ms / bin_ms
        half = int(np.ceil(3 * sigma))
        x = np.arange(-half, half + 1)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()
        rate = np.convolve(rate, kernel, mode="same")
    t = (edges[:-1] + edges[1:]) / 2.0
    return t, rate


def detect_peak(signal, fs, fmin, fmax):
    """Dominant frequency (Hz) and its power within [fmin, fmax] via Welch PSD."""
    from scipy.signal import welch
    sig = signal - np.mean(signal)
    nper = min(len(sig), int(fs * 8))  # up to 8 s windows
    nper = max(64, nper)
    f, pxx = welch(sig, fs=fs, nperseg=nper)
    band = (f >= fmin) & (f <= fmax)
    if not np.any(band):
        return 0.0, 0.0, f, pxx
    k = np.argmax(pxx[band])
    return float(f[band][k]), float(pxx[band][k]), f, pxx


def band_power(signal, fs, fmin, fmax):
    from scipy.signal import welch
    sig = signal - np.mean(signal)
    f, pxx = welch(sig, fs=fs, nperseg=max(64, min(len(sig), int(fs * 4))))
    band = (f >= fmin) & (f <= fmax)
    return float(np.trapezoid(pxx[band], f[band])) if np.any(band) else 0.0


def bandpass_envelope(signal, fs, lo, hi):
    """Zero-phase band-pass filter + Hilbert amplitude envelope.

    Returns (filtered, envelope) -- the canonical spindle representation: the
    band-pass trace shows the ~13 Hz oscillation, the envelope shows the
    waxing/waning amplitude of individual spindles.
    """
    from scipy.signal import butter, filtfilt, hilbert
    sig = np.asarray(signal, float)
    sig = sig - sig.mean()
    nyq = fs / 2.0
    b, a = butter(3, [lo / nyq, hi / nyq], btype="band")
    # filtfilt needs len > 3*max(len(a),len(b)); guard short signals
    if len(sig) <= 3 * max(len(a), len(b)):
        return sig, np.abs(sig)
    filt = filtfilt(b, a, sig)
    env = np.abs(hilbert(filt))
    return filt, env


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------

def make_plot(spikes, signals, meta, slow_peak, spindle_peak, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram

    tstop = meta["tstop"]
    tc, rc = signals["cortex"]           # cortical LFP-proxy (slow wave), fs_c
    tt, rt = signals["thalamus"]         # thalamic LFP-proxy (spindles), fs_t
    fs_c, fs_t = signals["fs_cortex"], signals["fs_thal"]

    # spindle band-pass (9-16 Hz) trace + envelope of the thalamic signal
    spin_filt, spin_env = bandpass_envelope(rt, fs_t, 9.0, 16.0)
    # slow-wave envelope of cortex, normalised, to mark UP/DOWN windows
    rc_n = (rc - rc.min()) / max(1e-9, (rc.max() - rc.min()))

    fig, axes = plt.subplots(5, 1, figsize=(11, 13))

    # 1. raster by layer
    ax = axes[0]
    layers = [l for l in ["MGB", "nRT", "L4", "L23", "L5", "L6"] if l in spikes]
    for i, layer in enumerate(layers):
        st = spikes[layer]["times"]
        if len(st):
            ax.plot(st, np.full_like(st, i) + 0.02 * (np.random.rand(len(st)) - 0.5),
                    "|", ms=4, alpha=0.5)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)
    ax.set_xlim(0, tstop)
    ax.set_title("Spike raster by layer (UP/DOWN banding = 1 Hz slow oscillation)")

    # 2. slow wave + spindle envelope -> shows spindles NESTED on UP states
    ax = axes[1]
    ax.plot(tc, rc_n, color="C0", lw=1.0, label="cortex slow wave (UP/DOWN)")
    ax.fill_between(tc, 0, rc_n, color="C0", alpha=0.12)
    env_n = spin_env / max(1e-9, spin_env.max())
    ax2 = ax.twinx()
    ax2.plot(tt, env_n, color="C3", lw=1.1, label="spindle (11-15 Hz) envelope")
    ax2.set_ylim(0, 1.25)
    ax2.set_ylabel("spindle env.", color="C3")
    ax.set_xlim(0, tstop)
    ax.set_ylabel("slow wave", color="C0")
    ax.set_title("Spindle envelope peaks ride the slow-wave UP states (nesting)")
    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, loc="upper right", fontsize=8)

    # 3. zoom: band-pass 13 Hz trace over one slow cycle -> individual spindles
    ax = axes[2]
    win = (tt >= 0) & (tt <= min(2000.0, tstop))
    ax.plot(tt[win], spin_filt[win], color="C3", lw=0.9, label="thalamus 9-16 Hz")
    ax.plot(tt[win], spin_env[win], color="k", lw=1.0, alpha=0.7, label="envelope")
    ax.plot(tt[win], -spin_env[win], color="k", lw=1.0, alpha=0.7)
    ax.set_xlim(0, min(2000.0, tstop))
    ax.set_title(f"Thalamic spindle oscillation, band-pass (peak {spindle_peak:.1f} Hz) "
                 "- zoom 0-2000 ms")
    ax.set_xlabel("time (ms)")
    ax.legend(loc="upper right", fontsize=8)

    # 4. thalamic spectrogram (time-frequency view of the spindles)
    ax = axes[3]
    sig = rt - rt.mean()
    nper = min(len(sig), max(64, int(fs_t * 0.4)))
    f, tspec, Sxx = spectrogram(sig, fs=fs_t, nperseg=nper, noverlap=int(nper * 0.9))
    fmask = f <= 25
    ax.pcolormesh(tspec * 1000.0, f[fmask], np.log1p(Sxx[fmask]), shading="auto")
    ax.axhline(13.0, color="w", ls="--", alpha=0.7)
    ax.set_ylim(0, 25)
    ax.set_title("Thalamic spectrogram - 13 Hz spindle bursts gated to UP states")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("frequency (Hz)")

    # 5. PSDs: cortical (slow) and thalamic (spindle) peaks
    ax = axes[4]
    _, _, fc, pc = detect_peak(rc, fs_c, 0.2, 4.0)
    _, _, ft, pt = detect_peak(rt, fs_t, 5.0, 25.0)
    ax.semilogy(fc, pc / max(1e-12, pc.max()), color="C0", label="cortex (slow)")
    ax.semilogy(ft, pt / max(1e-12, pt.max()), color="C3", label="thalamus (spindle)")
    ax.axvline(1.0, color="C0", ls="--", alpha=0.5)
    ax.axvline(13.0, color="C3", ls="--", alpha=0.5)
    ax.set_xlim(0, 25)
    ax.set_ylim(1e-4, 2)
    ax.set_title(f"PSD - slow-wave peak {slow_peak:.2f} Hz, spindle peak {spindle_peak:.1f} Hz")
    ax.set_xlabel("frequency (Hz)")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"Saved plot to {out_png}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=str, default=None,
                    help="YAML/JSON network config (default: built-in local sizes)")
    ap.add_argument("--tstop", type=float, default=None,
                    help="override simulation duration (ms)")
    ap.add_argument("--threads", type=int, default=1,
                    help="NEST local_num_threads (map to --cpus-per-task on MN5)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--outdir", type=str, default="out")
    ap.add_argument("--tag", type=str, default=None,
                    help="output filename tag (default: derived from config)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--no-assert", action="store_true",
                    help="do not exit non-zero if a rhythm is missing")
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = NetworkConfig.from_file(args.config) if args.config else NetworkConfig()
    if args.tstop is not None:
        cfg.tstop = args.tstop

    sim_config = SimulationConfig(num_threads=args.threads, seed=args.seed,
                                  record_traces=True, verbose=False)

    print(cfg.summary())
    print(f"Running NEST simulation ({cfg.tstop} ms, {args.threads} thread(s))...")

    model = AuditoryThalamoCorticalSleep(network_config=cfg, syn=SynapseParams(),
                                         sleep=SleepParams(), sim_config=sim_config)
    spikes, traces, meta = model.run()

    # ----- build LFP-proxy signals -----
    # Cortex: coarser bin, more smoothing -> clean ~1 Hz slow wave.
    # Thalamus: fine 1 ms bin (fs 1000 Hz), light smoothing -> resolves the
    # ~13 Hz spindle for band-pass/Hilbert analysis.
    def merged_signal(layers, bin_ms, smooth_ms):
        all_t = np.concatenate([spikes[l]["times"] for l in layers if l in spikes]) \
            if any(l in spikes for l in layers) else np.array([])
        return population_rate(all_t, cfg.tstop, bin_ms=bin_ms, smooth_ms=smooth_ms)

    fs_c = 1000.0 / 5.0
    fs_t = 1000.0 / 1.0
    tc, rc = merged_signal(["L23", "L5", "L6"], bin_ms=5.0, smooth_ms=25.0)
    tt, rt = merged_signal(["MGB", "nRT"], bin_ms=1.0, smooth_ms=3.0)
    signals = {"cortex": (tc, rc), "thalamus": (tt, rt),
               "fs_cortex": fs_c, "fs_thal": fs_t}

    # ----- analyse rhythms (discard the first 500 ms transient) -----
    rc_a = rc[int(500 / 5.0):] if len(rc) > int(500 / 5.0) else rc
    rt_a = rt[int(500 / 1.0):] if len(rt) > int(500 / 1.0) else rt

    slow_peak, slow_pow, _, _ = detect_peak(rc_a, fs_c, 0.3, 2.5)
    spindle_peak, spindle_pow, _, _ = detect_peak(rt_a, fs_t, 9.0, 16.0)
    spindle_band = band_power(rt_a, fs_t, 11.0, 15.0)

    print("\n--- results ---")
    for layer in ["MGB", "nRT", "L4", "L23", "L5", "L6"]:
        if layer in spikes:
            n = meta["n_per_layer"].get(layer, 0)
            nsp = len(spikes[layer]["times"])
            rate = 1000.0 * nsp / (cfg.tstop * max(1, n))
            print(f"  {layer:4s}: {nsp:6d} spikes  (~{rate:5.1f} Hz/neuron)")
    print(f"\n  slow-wave peak    : {slow_peak:.2f} Hz  (target ~1 Hz)")
    print(f"  spindle peak      : {spindle_peak:.2f} Hz  (target ~13 Hz)")
    print(f"  spindle-band power: {spindle_band:.3g} (11-15 Hz)")

    tag = args.tag or (Path(args.config).stem.replace("network_auditory_", "")
                       if args.config else "default")
    if not args.no_plot:
        make_plot(spikes, signals, meta, slow_peak, spindle_peak,
                  outdir / f"tc_sleep_{tag}.png")

    # ----- self-validation -----
    slow_ok = 0.5 <= slow_peak <= 1.8
    spindle_ok = 10.0 <= spindle_peak <= 16.0 and spindle_band > 0
    print(f"\n  slow-wave detected: {slow_ok}   spindle detected: {spindle_ok}")
    if not args.no_assert and not (slow_ok and spindle_ok):
        print("ERROR: expected rhythms not detected.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
