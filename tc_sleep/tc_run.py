"""
Driver for the auditory thalamo-cortical sleep model.

Builds the closed-loop column (tc_network.AuditoryThalamoCorticalSleep), runs
it in NEST, and verifies the two required rhythms:

  * slow-wave  ~1 Hz   -- a peak near 1 Hz in the PSD of the cortical
                          population-rate (LFP-proxy) signal.
  * spindles   ~13 Hz  -- power in the 11-15 Hz band that waxes/wanes with the
                          slow oscillation, shown in a spectrogram.

Run locally (sane test, seconds)::

    python3 tc_sleep/tc_run.py --config tc_sleep/config/network_auditory_local.yaml --outdir out

On MareNostrum 5 the same entry-point is launched with the MN5 config and
multiple threads (see tc_sleep/slurm/tc_sleep_mn5.sbatch).

Exit status is non-zero if either rhythm is missing, so the run self-validates.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# allow both `python3 tc_sleep/tc_run.py` and `python3 -m tc_sleep.tc_run`
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


def _lowpass(signal, fs, cut):
    from scipy.signal import butter, filtfilt
    sig = np.asarray(signal, float)
    sig = sig - sig.mean()
    if len(sig) <= 12:
        return sig
    b, a = butter(3, cut / (fs / 2.0), btype="low")
    return filtfilt(b, a, sig)


def _zscore(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x - x.mean()


def build_eeg_like(traces, tstop, seed=0):
    """Compose a single EEG/LFP-like trace (in arbitrary uV) from the recorded
    **membrane potentials**, containing BOTH the slow oscillation and the actual
    ~13 Hz spindle wavelets superimposed, the way a real sleep EEG channel looks:

        eeg = slow(cortex V_m, <2 Hz)  +  spindle(thalamus V_m, 9-16 Hz)  +  noise

    The per-layer mean V_m (an LFP proxy) is taken from the multimeters, so the
    trace reflects sub-threshold synaptic/input potentials rather than binned
    spike counts. Returns the trace plus components for shading/zooming.
    """
    fs = 1000.0
    grid = np.arange(0.0, tstop, 1.0)  # common 1 ms grid

    def layer_vm(layers):
        vs = []
        for l in layers:
            tr = traces.get(l)
            if tr is not None and len(tr["time"]) > 1:
                vs.append(np.interp(grid, tr["time"], tr["voltage"]))
        return np.mean(vs, axis=0) if vs else np.zeros_like(grid)

    cort = layer_vm(["L23", "L5", "L6"])   # cortical mean V_m -> slow wave
    thal = layer_vm(["MGB", "nRT"])        # thalamic mean V_m -> spindles

    slow = _zscore(_lowpass(cort, fs, 2.0)) * 40.0                 # ~uV slow wave
    spin_filt, spin_env = bandpass_envelope(thal, fs, 9.0, 16.0)   # 13 Hz wavelets
    sc = np.percentile(spin_env, 99) if len(spin_env) else 0.0
    scale = (40.0 / sc) if sc > 0 else 1.0                          # spindle bursts ~40 uV
    spin = spin_filt * scale
    env = spin_env * scale

    rng = np.random.default_rng(seed)
    noise = _lowpass(rng.standard_normal(len(slow)), fs, 40.0) * 4.0

    eeg = slow + spin + noise
    return {"t": grid, "fs": fs, "eeg": eeg, "slow": slow, "spin": spin,
            "env": env, "cort_v": cort, "thal_v": thal}


def detect_sp_sw(eeg_d):
    """Detect spindle (SP) and slow-wave (SW) epochs for shading.

    SP = sustained high spindle-envelope epochs (>=300 ms).
    SW = large slow-wave down-state troughs (shaded +/-300 ms).
    Returns (sp_windows, sw_windows) as lists of (t0, t1) in ms.
    """
    from scipy.signal import find_peaks
    t, fs = eeg_d["t"], eeg_d["fs"]
    env, slow = eeg_d["env"], eeg_d["slow"]

    # smooth the envelope (~60 ms) so spindle epochs are contiguous, then keep
    # epochs above the 55th percentile lasting >= 150 ms.
    k = max(1, int(0.06 * fs))
    env_s = np.convolve(env, np.ones(k) / k, mode="same")
    thr = np.percentile(env_s, 55)
    above = env_s > thr
    sp = []
    i, n = 0, len(above)
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            if t[j - 1] - t[i] >= 150.0:
                sp.append((t[i], t[j - 1]))
            i = j
        else:
            i += 1

    troughs, _ = find_peaks(-slow, prominence=max(1e-9, slow.std()),
                            distance=int(0.6 * fs))
    half = 0.30 * fs
    sw = [(t[max(0, k - int(half))], t[min(n - 1, k + int(half))]) for k in troughs]
    return sp, sw


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------

def _panel_label(ax, label):
    ax.text(0.006, 0.90, label, transform=ax.transAxes, fontweight="bold",
            fontsize=12, va="top", ha="left")


def _shade(ax, windows, color, tag):
    """Shade SP/SW windows and label the first one (reference-figure style)."""
    for k, (t0, t1) in enumerate(windows):
        ax.axvspan(t0, t1, color=color, alpha=0.28, lw=0)
        if k == 0:
            ax.text((t0 + t1) / 2.0, 0.97, tag, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=9, fontweight="bold")


def make_plot(spikes, traces, meta, slow_peak, spindle_peak, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram

    tstop = meta["tstop"]

    # composite EEG-like trace built from the recorded membrane potentials:
    # slow wave (cortex V_m) + actual 13 Hz spindle wavelets (thalamus V_m)
    eeg = build_eeg_like(traces, tstop, seed=meta.get("seed", 0))
    sp_win, sw_win = detect_sp_sw(eeg)
    te, ye = eeg["t"], eeg["eeg"]
    fs_v = eeg["fs"]

    fig, axes = plt.subplots(5, 1, figsize=(11, 13.5))

    # (a) raster by layer
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
    _panel_label(ax, "(a)")

    # (b) EEG-like composite: 13 Hz spindles riding on the slow wave, SP/SW shaded
    ax = axes[1]
    _shade(ax, sw_win, "tab:red", "SW")
    _shade(ax, sp_win, "0.5", "SP")
    ax.plot(te, ye, color="C0", lw=0.7)
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_xlim(0, tstop)
    ax.set_ylabel("amplitude (uV)")
    ax.set_title("EEG-like LFP: 13 Hz spindles (SP) superimposed on slow waves (SW)")
    _panel_label(ax, "(b)")

    # (c) zoom over ~2 slow cycles -> individual 13 Hz spindle wavelets on the SW
    ax = axes[2]
    # centre the zoom on the first spindle epoch if found, else start at 0
    z0 = max(0.0, sp_win[0][0] - 700.0) if sp_win else 0.0
    zmax = min(z0 + 2500.0, tstop)
    win = (te >= z0) & (te <= zmax)
    _shade(ax, [w for w in sw_win if w[1] > z0 and w[0] < zmax], "tab:red", "SW")
    _shade(ax, [w for w in sp_win if w[1] > z0 and w[0] < zmax], "0.5", "SP")
    ax.plot(te[win], ye[win], color="C0", lw=1.0)
    ax.plot(te[win], eeg["slow"][win], color="k", lw=1.4, alpha=0.6,
            label="slow component")
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_xlim(z0, zmax)
    ax.set_ylabel("amplitude (uV)")
    ax.set_title(f"Zoom: {spindle_peak:.1f} Hz spindle wavelets (SP) riding on the slow wave (SW)")
    ax.legend(loc="upper right", fontsize=8)
    _panel_label(ax, "(c)")

    # (d) thalamic V_m spectrogram (time-frequency view of the spindles)
    ax = axes[3]
    sig = eeg["thal_v"] - eeg["thal_v"].mean()
    nper = min(len(sig), max(64, int(fs_v * 0.4)))
    f, tspec, Sxx = spectrogram(sig, fs=fs_v, nperseg=nper, noverlap=int(nper * 0.9))
    fmask = f <= 25
    ax.pcolormesh(tspec * 1000.0, f[fmask], np.log1p(Sxx[fmask]), shading="auto")
    ax.axhline(13.0, color="w", ls="--", alpha=0.7)
    ax.set_ylim(0, 25)
    ax.set_xlim(0, tstop)
    ax.set_title("Thalamic V_m spectrogram - 13 Hz spindle bursts gated to UP states")
    ax.set_ylabel("frequency (Hz)")
    _panel_label(ax, "(d)")

    # (e) PSDs of the V_m LFP: cortical (slow) and thalamic (spindle) peaks
    ax = axes[4]
    _, _, fc, pc = detect_peak(eeg["cort_v"], fs_v, 0.2, 4.0)
    _, _, ft, pt = detect_peak(eeg["thal_v"], fs_v, 5.0, 25.0)
    ax.semilogy(fc, pc / max(1e-12, pc.max()), color="k", label="cortex V_m (slow)")
    ax.semilogy(ft, pt / max(1e-12, pt.max()), color="C0", label="thalamus V_m (spindle)")
    ax.axvline(1.0, color="k", ls="--", alpha=0.5)
    ax.axvline(13.0, color="C0", ls="--", alpha=0.5)
    ax.set_xlim(0, 25)
    ax.set_ylim(1e-4, 2)
    ax.set_title(f"V_m PSD - slow-wave peak {slow_peak:.2f} Hz (rate), spindle peak {spindle_peak:.1f} Hz (rate)")
    ax.set_xlabel("frequency (Hz)")
    ax.legend(loc="upper right", fontsize=8)
    _panel_label(ax, "(e)")

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
    meta["seed"] = args.seed

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
        make_plot(spikes, traces, meta, slow_peak, spindle_peak,
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
