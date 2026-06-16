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
python pd_phase1_bgth_network.py
```

This prints approximate population firing rates per nucleus for the
healthy, PD, and PD+DBS conditions, and saves a comparison plot to
`pd_phase1_bgth_output.png`.

## Phase 2 prototype: motor cortex integration

`pd_phase2_motor_cortex.py` extends Phase 1 with a primary motor cortex
(M1) population driven by the thalamus (Th -> M1), mirroring bigMC's
chaining of a basal-ganglia stage into a motor cortex stage. Run with:

```bash
python pd_phase2_motor_cortex.py
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
python pd_phase3_spinal_force.py
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
python pd_phase4_spinal_cpg.py
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
