# Data requirements for personalizing the PD model (IRCCS Neurolesi)

This document specifies the data we would need from **IRCCS Centro Neurolesi
"Bonino Pulejo"** to (a) personalize the basal-ganglia–thalamus–cortex–spinal
model in this repository to individual patients and (b) make it capable of
**treatment-effect prediction**.

It is grounded in the IRCCS study *"Brain connectivity fingerprinting as a
predictive biomarker of art therapy outcomes in Parkinson's disease"*
(Ielo et al., Research Square, 2026; DOI 10.21203/rs.3.rs-8475840/v1;
ClinicalTrials.gov NCT03178786), which defines what the centre already
measures.

> **Two prediction targets, two data needs.** The published trial endpoint is
> an **art-therapy** outcome (ΔUPDRS-III). Our biophysical model is most
> naturally predictive of **DBS / dopaminergic** outcomes. The art-therapy
> data lets us *explain* the published connectome-fingerprint finding
> mechanistically; the DBS data (mostly **not** in the current paper) is what
> makes the model a treatment-effect predictor in its own right. Both are
> listed below.

---

## A. Data the centre already collects (per the paper's Methods)

These map directly onto model parameters / validation targets.

| Item | Detail (from paper) | Model use |
|---|---|---|
| **UPDRS-III**, pre and post | mean 37.8 → 32.5; responder = ≥15% reduction | Ground-truth outcome label; global motor severity |
| **Hoehn & Yahr stage** | inclusion 2–3 | Sets dopamine-depletion / indirect-pathway gain |
| **MoCA**, **BDI-II** | MoCA > 22, BDI-II < 20 | Covariates; cognitive/affective network state |
| **Demographics** | age, sex, education | Matching, covariates |
| **Dopaminergic regimen** | stable during study; scanned ON | Baseline dopamine tone (need LEDD value, see B) |
| **Structural MRI** | 3D T1 MPRAGE 1 mm; axial T2 | Anatomy; ROI definition |
| **Resting-state fMRI** | eyes-closed, 400 vol (~5.7 min), 2 mm, ON state | Empirical functional connectome (FC) target |
| **Parcellation** | Schaefer-2018, 400 regions / 7 networks | Region scheme for connectome + FC fitting |
| **Functional connectome (FC)** | 400×400 Pearson FC, residualized | Per-subject FC to fit macroscale coupling |
| **Fingerprint metrics** | I_self, I_others, I_diff, I_diff-norm, SR, ICC | Validation of subject identifiability |
| **Graph topology measures** | eigenvector centrality (top predictor), assortativity, rich-club, strength, mean first-passage time | Validation features; link to model network topology |

**Minimal request for mechanistic/explanatory work (art therapy):**
de-identified subject-level table (UPDRS-III pre/post + responder label,
H&Y, MoCA, BDI-II, age/sex/education) **plus** the per-subject Schaefer-400
FC matrices (or the raw rs-fMRI), **plus** T1/T2.

---

## B. Additional data needed for treatment-effect prediction

Items the current paper does **not** report and that we would need to
request explicitly.

### B1. Structural connectivity (for personalized coupling)
- **Diffusion MRI (DWI/DTI)**: multi-shell preferred; b-values, directions,
  and a reverse-phase-encode b0 for distortion correction.
  *Why:* tractography → subject **structural connectome** (edge weights +
  tract lengths) → personalized network coupling **and conduction delays**.
  The paper acquires T1/T2 but does not mention DWI.

### B2. DBS / stimulation data (if DBS response is a target)
- **Target & lead localization**: STN vs GPi; electrode contact coordinates
  (post-op CT/MRI) in subject and MNI space.
- **Stimulation parameters**: active contact(s), amplitude, frequency, pulse
  width; monopolar/bipolar configuration.
- **Pre/post-DBS UPDRS-III**, ideally in the standard conditions
  (med-OFF/stim-OFF, med-OFF/stim-ON, med-ON/stim-ON).
- **LEDD** (levodopa-equivalent daily dose) pre/post.

### B3. Motor / tremor quantification (to validate Phase 4)
- **Accelerometry or surface EMG** of the affected limb at rest and during
  posture/action (tremor frequency 3–7 Hz and amplitude), **or** at minimum
  the **UPDRS-III item-level tremor scores** (items 3.15–3.18) and
  bradykinesia items.
  *Why:* our simulated 4–6 Hz limb-force tremor and bradykinesia
  (voluntary-drive collapse) need a measured counterpart to fit/validate.

### B4. Longitudinal time points (for progression / trajectory modeling)
- Repeated clinical + imaging assessments over time (the paper notes
  longitudinal designs as future work).

---

## C. How each item enters the personalization pipeline

```
T1/T2 (+DWI)            -> structural connectome  -> coupling weights & delays
rs-fMRI                 -> empirical FC           -> tune global coupling (virtual patient)
UPDRS-III / H&Y / LEDD  -> dopamine depletion     -> indirect-pathway / GPi->Th gain
DBS params (B2)         -> stimulation model      -> simulate candidate therapy
accelerometry/EMG (B3)  -> tremor spectrum        -> validate Phase 4 limb force
outcome (ΔUPDRS-III)    -> held-out label         -> evaluate prediction
```

1. **Anatomy → network.** Structural connectome sets per-subject coupling
   weights and conduction delays (replaces the hand-set `PROJECTIONS`).
2. **Function → fit.** Tune global coupling so simulated FC matches the
   subject's empirical Schaefer-400 FC ("virtual patient").
3. **Clinical → state.** UPDRS-III / H&Y / LEDD set the dopamine-depletion
   level (indirect-pathway and GPi→Th gains).
4. **Simulate treatment.** Apply candidate DBS target/parameters (or a
   dopaminergic level) and read out predicted ΔUPDRS-III / Δtremor.
5. **Validate.** Compare against held-out post-treatment outcome; quantify
   with the same responder definition (≥15% UPDRS-III reduction) and ROC-AUC
   used in the paper.

---

## D. Format, ethics, and logistics

- **De-identification / BIDS.** Imaging in **BIDS** (defaced T1/T2/DWI,
  rs-fMRI + JSON sidecars); tabular clinical data as CSV with a documented
  data dictionary; subject IDs pseudonymized with a key held at IRCCS.
- **Derivatives are often enough.** For much of the work, the Schaefer-400
  FC matrices + (if available) structural connectomes + clinical CSV suffice
  and reduce privacy exposure vs. raw imaging.
- **Governance.** Data-sharing/processing agreement (GDPR), ethics/IRB
  coverage referencing NCT03178786, and confirmation of consent scope for
  modeling/secondary use.
- **Provenance.** fMRIPrep / XCP-D versions and parameters (the paper uses
  fMRIPrep 24.1.1 and XCP-D) so our preprocessing matches theirs.

---

*See `pd_phase5_hpc_sweep.py` for the compute side: the same biomarkers
listed here (population rates, tremor spectrum) are what the HPC parameter
sweep / inference fits to these measured quantities.*
