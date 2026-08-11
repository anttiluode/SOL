# SOL — inventions by Sol

This repository is the **useful-things branch** of the Geometric Neuron line.

The rule is deliberately different from the theory repos:

> **Take the surviving mechanisms, combine them, and make small runnable things that either work or fail clearly.**

No claim here is new merely because two old ideas were joined. No benchmark is a victory until it survives controls. The point of SOL is to turn the long GeometricNeuron → TWC → KYY walk into **demos, instruments, audits, compilers, and candidate components**.

## What came in from the parent projects

From **GeometricNeuron_V20**:

- the skew lag operator `A_tau=(C_tau-C_tau.T)/2` as a readout of temporal ordering;
- the useful engineering version of "energy on surprise": update the expensive/fast state when prediction fails;
- the discipline of keeping killed ideas in the ledger.

From **KYY**:

- a recurrent body made from products of **local reciprocal orthogonal two-port scatterers**;
- geometry as an explicit sparse constraint rather than a story about generic structured recurrence;
- behavioral relations can matter more than reproducing an arbitrary learned matrix;
- conservative/lossless state can **hide** a distinction without actually forgetting it;
- exact digital behavior can have many equivalent state geometries with very different physical margins/costs;
- strong generic controls remain mandatory.

From **TransientWaveCompiler**:

- exact local sensitivities;
- nuisance/physics separation and parameter diagnosis;
- identifiability as a first-class output;
- when the current measurement is ambiguous, **change the experiment instead of optimizing harder**.

The SOL inventions are what happens when those pieces are allowed to touch.

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

KYY's local scatter recurrence is used as the persistent state mixer. A learned predictor estimates the incoming continuous sample; only the residual is proposed for writing, and the write gate grows with surprise. The demo is a noisy piecewise-constant sensor tracking problem and reports MSE, hidden-level accuracy, mean write gate, parameter count, and GRU / always-write controls.

```bash
python 03_surprise_scatter_nn.py --steps 250
```

**This is not yet claimed to be a better RNN.** The interesting hypothesis is narrower: can an orthogonal local state body retain useful memory while **writing much less often** on predictable streams? If the GRU or always-write model dominates after counting real compute and writes, we keep the null.

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

## SOL-05 — Zombie Memory Audit

**Test whether a claimed reset/unlearning event really destroyed history or merely hid it from the current output.**

Two histories can be output-identical at the reset instant while remaining different in hidden state. `SOL-05` breadth-first searches a **common future suffix**: the same later inputs are applied to both histories, looking for a word that rotates the hidden difference back into the readout.

```bash
python 05_zombie_memory_audit.py
```

The built-in control compares:

- a lossless fake reset that moves history into a readout blind spot;
- a singular `pinch` that actually collapses the two hidden states.

The fake reset is exposed immediately; the destructive reset cannot resurrect information that is no longer present.

Possible use: a small adversarial audit for RNN reset semantics, state compression, privacy/unlearning experiments, or any recurrent controller where **"the outputs match now" is being used as evidence of forgetting**.

Boundary: searching a finite suffix alphabet does not prove information-theoretic erasure in an arbitrary nonlinear system. It is a constructive falsifier: finding one exposing suffix is enough to disprove the reset claim.

---

## SOL-06 — Behavior / Gauge Doctor

**Self-calibrate the behavior you require without pretending an unobservable internal coordinate is identifiable.**

Two reciprocal reflection cells form a tiny noncommutative machine. Their required relation is

```text
s^2 = I
 t^2 = I
(st)^3 = I
```

The relation constrains only the **relative** angle of the two cells. A common drift of both angles is a genuine gauge direction: the algebra remains exact while the external port moves.

`SOL-06` first performs relation-only repair and deliberately shows the failure: algebraic defect goes to numerical zero, but the port remains wrong. It then automatically chooses one scalar probe/port anchor that completes the Jacobian rank and repairs the observable behavior too.

```bash
python 06_behavior_gauge_doctor.py
```

This is a direct merger of KYY's "compile behavioral relations, not matrices" idea with TWC's identifiability discipline.

Potential use: self-calibration of phase/oscillator networks, structured recurrent hardware, or any system where multiple internal realizations are behaviorally equivalent and restoring an arbitrary historical matrix would be the wrong objective.

---

## SOL-07 — Phase Handoff Compiler

**Optimize the geometry of a whole analog/quantized pipeline, not each stage in isolation.**

A `C4 -> C2 -> C4` phase pipeline merges neighboring phase states and later restores four-way locking. The locally obvious first-stage design places the coarse well at the midpoint `alpha=pi/4`. That maximizes capture margin for the merge — and puts the state **exactly on the next stage's separatrix**.

The compiler instead maximizes the minimum margin across capture *and* restoration, selecting `alpha=pi/8`.

```bash
python 07_phase_handoff_compiler.py
```

At the default `0.08 rad` simulated phase noise, the smoke run gives approximately:

```text
alpha = pi/8    end-to-end accuracy ~100%
alpha = pi/4    end-to-end accuracy ~50%
```

This is a reduced phase/quantizer model, not a hardware result. The wider design rule is the point: **a stage can be locally perfect yet hand the next stage a geometrically disastrous state. Compile the composition.**

Possible uses: oscillator logic, relocking pipelines, cascaded quantizers, analog/digital interfaces, and structured NN/hardware compilers where intermediate state geometry matters.

---

## Install / smoke test

All demos except SOL-03 need only NumPy. SOL-03 also needs PyTorch.

```bash
pip install -r requirements.txt
python test_sol.py
```

See `FIRST_SMOKE_RESULTS.md` for the current development-machine numbers and negative controls.

## The direction from here

The strongest seam is becoming **behavior-aware self-diagnosing computation**:

```text
local geometric recurrence
        +
exact sensitivity of outputs to named local elements
        +
active choice of probe when those elements are ambiguous
        +
behavioral relations instead of arbitrary matrix identity
        +
explicit destructive operations where forgetting is required
        +
composition-aware state geometry
```

That is more concrete than "a geometric neuron chip." It suggests machines that compute normally, know which distinctions must survive or disappear, know when their measurements cannot identify an internal cause, and can deliberately probe/calibrate themselves.

The NN branch (`SOL-03`) remains the place to attack the "better NN" hope. The immediate controls should be GRU, strong Householder/DeltaProduct-style recurrence, KYY `geom_scatter`, and a parameter/compute-matched event-gated baseline. A useful win condition is not a toy parity score: **accuracy/robustness per write, per local operation, under reset pressure, or under quantization/faults on real streams.**

Do not hype. Do not lie. **Make things and measure them.**
