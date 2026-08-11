# SOL-08 — Depth Spectrometer

## The accident

`AnttisBrain2/VersionWithComputeRecursionDepthUsingZandXButtons.html` does something stranger than ordinary early exit.

Each room has a native computation depth `K`. The player changes rendered iteration depth `n` with Z/X. At `n=K` the room reveals its native form plus transient filigree; away from `K` the rendered geometry collapses toward the same dull spherical attractor. The collision geometry deliberately remains the true room geometry.

So the game accidentally treats **iteration depth as a probe coordinate**. The informative object is not only the converged state; it can be the transient path toward it.

This is the inverse-side warning to Horizon Net:

```text
Horizon Net:
    stop once more iterations cannot change the answer.

Depth Spectrometer:
    do not assume the fixed point contains all information;
    some distinctions may exist only at finite iteration depth.
```

## Minimal kill test

Use six stable two-dimensional machines

```text
x[n+1] = rho R(theta) x[n]
x[0]   = (1,0)
y[n]   = first coordinate of x[n]
```

with

```text
rho = 0.88
theta = [0.25, 0.28, 0.31, 0.34, 0.37, 0.40] rad
measurement noise sigma = 0.06
```

Every machine has exactly the same infinite-depth fixed point:

```text
x* = 0.
```

The scalar port is

```text
y[n] = rho^n cos(n theta).
```

So theta is visible only through the transient.

We permit exactly three noisy observations and search every distinct three-depth set from depths 1..40. The objective is the minimum pairwise squared template separation in noise units. There are only

```text
C(40,3) = 9880
```

candidate ears, so the demo uses exhaustive search rather than a heuristic optimizer.

## Result

The optimal depth set is

```text
(4, 5, 6)
```

with worst pairwise information score

```text
4.0210 noise^2 units.
```

Monte Carlo nearest-template identification:

```text
three selected transient depths (4,5,6):   76.05%
three repeats of final depth (40,40,40):   19.20%
first three depths (1,2,3):                47.64%
200 random three-depth sets, mean:         46.79%
200 random three-depth sets, median:       48.13%
random p10 .. p90:                         30.23% .. 61.78%
```

Six-way chance is 16.67%.

At depth 40 the contraction envelope is already

```text
rho^40 = 0.006016,
```

so the candidate machines have almost collapsed onto the same observable state.

The average scalar Fisher information for `theta` peaks at depth

```text
5.
```

The continuous sensitivity envelope `t rho^t` peaks near

```text
-1/log(rho) = 7.82,
```

with the exact discrete optimum shifted earlier by the phase terms and by the requirement to separate all six candidate thetas simultaneously.

## What survived

The useful statement is not "transients contain information"; that is classical system identification.

The useful compiler/instrument rule is:

> **Treat computation depth itself as an experimental control variable. If different hidden mechanisms share a fixed point, choose the finite iteration depths that maximize their observable separation instead of automatically reading only the converged state.**

This connects four older branches without requiring a new architecture:

```text
AnttisBrain2
    hidden native computation depth; informative transient

Visertaja
    trajectory can carry information discarded by a final-state readout

KYY zombie-memory / future observability
    a distinction absent at one port/time need not be behaviorally dead

TWC active probe
    choose the observation that makes hidden mechanisms identifiable
```

The combination suggests a **depth spectrometer** for weight-tied nets, DEQs, recurrent operators, iterative solvers, diffusion/sampling trajectories, or physical relaxation processes: record a chosen port across iterations, calculate where parameter/class/fault sensitivity actually lives, and select only those depths.

## What is not claimed

- Transient-response identification is old.
- Fisher-information experiment design is old.
- Early-exit / anytime inference is old.
- This reduced damped-rotation demo is not evidence of a better neural network.
- It does not show that intermediate depths are always better; it shows exactly the condition where they matter: **distinct transient operators with an observationally collapsed asymptote.**

The next honest test is not another synthetic oscillator. Apply the depth spectrometer to an existing iterative model whose final readout has already been shown to discard trajectory information, and ask whether actively selected depths recover useful information at lower readout cost.

Source: `08_depth_spectrometer.py`.
