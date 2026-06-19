# PD
the Parkinson's disease (PD) repository

## Phase 1 prototype: BGTH network

`pd_phase1_bgth_network.py` is a first prototype, inspired by the
phase-based structure of [bigMC](https://github.com/max-talanov/bigMC) and
the multiscale basal ganglia-thalamus (BGTH) model described in
[this article](https://pmc.ncbi.nlm.nih.gov/articles/PMC12458589/).

It simulates four coupled nuclei (STN, GPe, GPi, Thalamus), each a small
population of conductance-based Hodgkin-Huxley neurons connected via
mean-field excitatory/inhibitory synapses. It supports a "healthy" and a
"PD" parameter regime (altered coupling strengths and external drive), plus
an optional deep brain stimulation (DBS) mode.

DBS is modelled as functional silencing of the stimulated nucleus' efferent
output (stimulation-induced depolarization block / synaptic depression).
The default target is **GPi**: suppressing the over-active GPi → thalamus
inhibition lets thalamocortical output recover (Th rate PD ~58 Hz →
PD+DBS ~157 Hz), which is the therapeutic direction and propagates through
all downstream phases. (STN-DBS is far less effective in this rate-level
model — a known subtlety of firing-rate accounts of the basal ganglia — so
GPi, also a standard clinical target, is the default.)

### Run

```bash
pip install -r requirements.txt
python3 pd_phase1_bgth_network.py
```

This prints approximate population firing rates per nucleus for the
healthy, PD, and PD+DBS conditions, and saves a comparison plot to
`pd_phase1_bgth_output.png`.

## Phase 2 prototype: motor cortex integration

`pd_phase2_motor_cortex.py` extends Phase 1 with a primary motor cortex
(M1) population driven by the thalamus (Th -> M1), mirroring bigMC's
chaining of a basal-ganglia stage into a motor cortex stage. Run with:

```bash
python3 pd_phase2_motor_cortex.py
```

Saves `pd_phase2_motor_cortex_output.png` comparing STN/Th/M1 membrane
potentials across healthy, PD, and PD+DBS conditions.

## Phase 3 prototype: spinal cord force / tremor profile

`pd_phase3_spinal_force.py` extends Phase 2 with a simple spinal stage:
M1 output drives a leaky-integrator alpha-motoneuron pool, whose firing
rate is convolved with a muscle twitch kernel to produce a limb force
profile (bigMC's cortex -> spinal CPG -> limb progression). It reports
the dominant frequency in the 3-7 Hz Parkinsonian resting-tremor band and
saves `pd_phase3_spinal_force_output.png` (force traces + power spectra).

```bash
python3 pd_phase3_spinal_force.py
```

**Note:** Phase 3 reads the M1 *mean membrane potential*, which during
asynchronous tonic firing carries almost no coherent oscillation, so its
force spectrum does not show tremor. Seeing tremor requires rhythm
generators at the population level — see Phase 4.

## Phase 4 prototype: spinal CPG + visible limb tremor

`pd_phase4_spinal_cpg.py` is where the Parkinsonian tremor actually
becomes visible in a limb force profile. It adds a spinal central pattern
generator (CPG) — a Matsuoka flexor/extensor half-center oscillator, the
same reciprocal-inhibition-with-fatigue mechanism used by
[tinyCPG](https://github.com/max-talanov/tinyCPG) — plus a second Matsuoka
oscillator tuned to the 4-6 Hz tremor band.

Both rhythm generators are *driven by* the Phase 1-2 BGTH + M1 spiking
model:

- **voluntary CPG drive** ← thalamocortical output (thalamus firing rate).
  PD collapses thalamic firing → weak, slow voluntary force
  (bradykinesia).
- **tremor drive** ← thalamic suppression below the healthy reference
  (GPi over-inhibition → thalamic rebound bursting).

The DBS benefit here is **emergent**, not hand-tuned: GPi-DBS restores
thalamic firing in the spiking model (Phase 1), which simultaneously
raises the voluntary CPG drive (less bradykinesia) and lowers the tremor
drive — no separate DBS-tremor factor is applied in this phase.

The CPG drives flexor/extensor motoneuron pools in antagonist fashion
(it moves the limb); the tremor oscillation is injected common-mode (it
shakes the limb). Each pool is low-pass filtered and convolved with a
muscle twitch; the net joint force is their difference.

```bash
python3 pd_phase4_spinal_cpg.py
```

Reports voluntary/tremor drives, the tremor-band peak frequency, and
absolute tremor amplitude per condition, and saves
`pd_phase4_spinal_cpg_output.png` (force traces + spectra). Representative
result: healthy shows a slow voluntary rhythm with no tremor; PD shows a
sustained ~4-5 Hz rest tremor with collapsed voluntary movement; PD+DBS
restores the voluntary rhythm and suppresses the tremor amplitude
roughly two-fold.

### Why the rhythm generators live at the population/rate level

The mean-field Hodgkin-Huxley pools (Phases 1-2) fire tonically but
asynchronously, so they carry essentially no coherent tremor-band
oscillation; coercing a small spiking pool into a clean limit cycle is
brittle. Following the multiscale co-simulation philosophy of the source
article (microscale neurons + macroscale dynamics), Phase 4 keeps the
spiking network as the microscale layer and places the robust, tunable
rhythm generators (CPG + tremor) at the macroscale rate level, coupled to
the spiking model through firing-rate read-outs. A natural next step is to
close this loop with a genuinely oscillating (synchronised) STN-GPe
spiking sub-circuit.

## Phase 5: HPC parameter sweep (BSC MareNostrum 5 ready)

`pd_phase5_hpc_sweep.py` wraps the Phase 1-4 model in an embarrassingly
parallel parameter sweep so it can run as a SLURM job array on a
supercomputer (e.g. BSC MareNostrum 5) and, later, swap the small
mean-field backend for a large-scale spiking backend (NEST) behind the
same interface. Each array task runs one grid point and writes a small
JSON file; an aggregation step merges them into one CSV for downstream
parameter inference against measured biomarkers.

```bash
python3 pd_phase5_hpc_sweep.py --list                 # grid size
python3 pd_phase5_hpc_sweep.py --smoke --outdir out   # run first few points locally
python3 pd_phase5_hpc_sweep.py --aggregate --outdir out
```

On the cluster, `slurm/pd_phase5_sweep.sbatch` launches the array; see its
header for submission. The `nest` backend is a documented stub describing
the mean-field → spiking mapping to implement cluster-side.

## Thalamo-cortical sleep model (auditory) — NEST

`tc_sleep/` is a separate, full-spiking **NEST** model (the PD phases above are
mean-field). It builds a closed-loop **auditory** thalamo-cortical column and,
under sleep drives, reproduces the **slow oscillation (~1 Hz)** with **sleep
spindles (~13 Hz)** nested on its UP states — **no optimisation**.

The column architecture is adapted from the Cortical Column model
([max-talanov/cc, `optimization_nest`](https://github.com/max-talanov/cc/tree/main/optimization_nest)):
five stages of `iaf_cond_exp` neurons (thalamus TCR/nRT, L4, L2/3, L5, L6 with
RS/FRB/TuftRS/IB and Basket/LTS/Axoaxonic subtypes), `pairwise_bernoulli`
wiring. On top of cc it (1) **closes the loop** — adds corticothalamic feedback
`L6→thalamus` and reciprocal reticular inhibition `nRT→TCR` (cc has neither),
the TCR↔nRT loop being the spindle generator — and (2) adds a 1 Hz cortical
slow-wave drive plus a 13 Hz, UP-gated thalamic spindle drive.

```bash
# local sane test (seconds; needs NEST 3.x + numpy/scipy/matplotlib/pyyaml)
python3 tc_sleep/tc_run.py --config tc_sleep/config/network_auditory_local.yaml --outdir out
# bio-plausible run on BSC MareNostrum 5 (edit account/modules first)
sbatch tc_sleep/slurm/tc_sleep_mn5.sbatch
```

The driver prints the detected slow-wave / spindle peak frequencies, saves a
raster + LFP-proxy + PSD + spectrogram plot, and exits non-zero if either
rhythm is absent (self-validating). Local and MN5 configs share one code path;
synaptic weights are balanced-scaled by population size so the ~1150-neuron MN5
column keeps the same firing regime as the ~112-neuron local column. See
[`tc_sleep/README.md`](tc_sleep/README.md) for the full description, including
the honest-scope note (intrinsic-current spindles would need `ht_neuron`).

## Personalization & clinical data

`docs/DATA_REQUIREMENTS.md` specifies the data needed from IRCCS Centro
Neurolesi "Bonino Pulejo" to personalize the model to individual patients
and make it capable of treatment-effect prediction, grounded in the
centre's brain-fingerprinting art-therapy study (NCT03178786). It covers
what the centre already collects (clinical scores, T1/T2, rs-fMRI,
Schaefer-400 FC) and what would additionally be needed (diffusion MRI for
structural connectomes, DBS parameters, tremor accelerometry/EMG,
longitudinal time points), plus how each item enters the personalization
pipeline.
