# SOL — inventions by Sol

This repository is the **useful-things branch** of the Geometric Neuron line.

The rule is deliberately different from the theory repos:

> **Take the surviving mechanisms, combine them, and make small runnable things that either work or fail clearly.**

No claim here is new merely because two old ideas were joined. No benchmark is a victory until it survives controls. The point of SOL is to turn the long GeometricNeuron → TWC → KYY walk into **demos, instruments, and candidate components**.

## What came in from the parent projects

From **GeometricNeuron_V20**:

- the skew lag operator `A_tau=(C_tau-C_tau.T)/2` as a readout of temporal ordering;
- the useful engineering version of "energy on surprise": update the expensive/fast state when prediction fails;
- the discipline of keeping killed ideas in the ledger.

From **KYY**:

- a recurrent body made from products of **local reciprocal orthogonal two-port scatterers**;
- geometry as an explicit sparse constraint rather than a story about generic structured recurrence;
- strong generic controls remain mandatory.

From **TransientWaveCompiler**:

- exact local sensitivities;
- nuisance/physics separation and parameter diagnosis;
- identifiability as a first-class output;
- when the current measurement is ambiguous, **change the experiment instead of optimizing harder**.

The first SOL inventions are what happens when those three are allowed to touch.

---

## SOL-01 — Edge Doctor

**A recurrent/local-wave network that can diagnose its own drifting edge.**

A hidden angle error is inserted into one local scatter element. The demo observes only two output ports under a handful of transient pings, propagates the **exact Jacobian of the output with respect to every physical edge**, and ranks the fault.

```bash
python 01_edge_doctor.py
```

Why it could matter: if a geometric recurrent network is ever mapped to analog, photonic, RF, switched-cap, memristive, or aggressively quantized hardware, the geometry gives every local element a name. TWC-style sensitivity can make the model **calibratable and fault-localizable** instead of opaque.

Falsifier: if realistic noise/nonlinearity makes faults non-identifiable or generic black-box calibration does better at the same measurement cost, the advantage is gone.

---

## SOL-02 — Active Probe

**Let the model choose the next ping that best separates possible internal faults.**

Candidate injections are scored by the information in their response Jacobians. A greedy selector builds a probe set that raises observability of the local edge parameters, then a Monte Carlo fault test compares active, random, and repeated probes.

```bash
python 02_active_probe.py
```

This is the most direct child of TWC's identifiability work: an ambiguous model should say **"I need a different experiment"** and propose one.

Possible uses: self-test in structured neural hardware, acoustic/ultrasonic fixtures, resonator networks, sensor calibration, robotics where a system can deliberately twitch/actuate itself to identify what changed.

---

## SOL-03 — Surprise Scatter Cell

**Candidate NN cell: local orthogonal memory + prediction-error-gated write.**

KYY's local scatter recurrence is used as the persistent state mixer. A learned predictor estimates the incoming continuous sample; only the residual is proposed for writing, and the write gate grows with surprise. The demo is a noisy piecewise-constant sensor tracking problem and reports:

- MSE;
- discrete hidden-level accuracy;
- mean state-write gate;
- parameter count;
- GRU and always-write scatter controls.

```bash
python 03_surprise_scatter_nn.py --steps 250
```

**This is not yet claimed to be a better RNN.** The interesting hypothesis is narrower: can an orthogonal local state body retain useful memory while **writing much less often** on predictable streams? If yes, that may matter for event-driven/low-write/physical implementations. If the GRU or always-write model dominates after counting real compute and writes, we keep the null.

---

## SOL-04 — Arrow Monitor

**Directional anomaly detection for multichannel time series.**

The GeometricNeuron V9 skew operator is stripped of the neuron interpretation and used as an instrument. A reference window learns the lag-skew matrix; later windows are compared to it. A process can keep roughly the same per-channel spectrum and amplitude while the **ordering/circulation between channels reverses**.

```bash
python 04_arrow_monitor.py
python 04_arrow_monitor.py --csv your_multichannel_numeric.csv --tau 1
```

Possible uses: rotating machinery, process sensors, multichannel audio, motion sensors, physiological research signals. It is an anomaly fingerprint, **not a causal-discovery claim**.

---

## Install / smoke test

The first, second and fourth demos only need NumPy. SOL-03 also needs PyTorch.

```bash
pip install -r requirements.txt
python test_sol.py
```

## The direction from here

The most promising new seam is **self-diagnosing computation**:

```text
local geometric recurrence
        +
exact sensitivity of outputs to named local elements
        +
active choice of probe when those elements are ambiguous
        +
surprise-triggered state write
```

That is already more concrete than "a geometric neuron chip." It suggests a machine that computes normally, knows which local degrees of freedom its behavior depends on, and can deliberately probe itself when something drifts.

The NN branch (`SOL-03`) is the place to attack the "better NN" hope. The immediate controls should be GRU, strong Householder/DeltaProduct-style recurrence, KYY `geom_scatter`, and a parameter/compute-matched event-gated baseline. The useful win condition is not a toy parity score: **accuracy/robustness per write, per local operation, or under quantization/faults on real streams.**

Do not hype. Do not lie. **Make things and measure them.**
