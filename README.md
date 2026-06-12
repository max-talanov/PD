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
an optional high-frequency DBS pulse train applied to STN.

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

**Known limitation of this first pass:** with the current coupling
strengths, the Th -> M1 drive is too weak relative to M1's own dynamics
for the PD beta/tremor-band oscillations to clearly show up in the force
spectrum (healthy/PD/PD+DBS spectra come out nearly identical). Future
iterations should strengthen the thalamocortical projection and/or
amplify low-frequency STN/GPe oscillations before they reach M1.
