# First smoke results — 2026-08-11

These are development-machine smoke checks, not benchmark claims.

## SOL-01 Edge Doctor

Default demo parameters, NumPy-only:

- hidden drift: edge 5, +0.06000 rad;
- 8 local edge parameters, two observed output ports, four transient probes;
- response Jacobian rank: 8/8;
- exact analytic Jacobian vs finite difference max error: 1.956e-05;
- Gauss-Newton top-1 diagnosis: edge 5 — PASS;
- recovered drift: +0.058766 rad in that noisy run.

## SOL-02 Active Probe

One 8-edge nominal network, 4 selected pings, measurement noise sigma=8e-4, 120 Monte Carlo fault trials:

- active probe nodes: [1, 0, 5, 6]; rank 8/8; information score +13.52;
- one random four-node set: rank 8/8; score +9.38;
- repeating node 0 four times: rank 2/8; score -75.15;
- top-1 fault localization: active 100.0%, random 100.0%, repeated probe 30.8%.

The active-vs-random hit rate is therefore **not** yet a demonstrated advantage at this noise point; the clean result is the identifiability result: repeated probing is rank-deficient, while informative probe diversity restores full rank. Stronger noise and fewer probes are the next honest comparison.

## SOL-03 Surprise Scatter Cell

CPU, state_dim=16, 120 training steps, seed 1:

| model | MSE | hidden-level accuracy | mean write gate | trainable params |
|---|---:|---:|---:|---:|
| surprise-scatter | 0.03361 | 93.2% | 0.603 | 133 |
| always-write scatter | 0.03883 | 91.7% | 1.000 | 133 |
| GRU | 0.04405 | 93.3% | 1.000 | 929 |

A shorter 80-step seed-2 run did **not** preserve the MSE ordering: surprise-scatter MSE 0.05743 vs always-write 0.04874 and GRU 0.05047, while hidden-level accuracy remained 92.1%, 90.6%, and 92.5% respectively; mean surprise write gate was 0.615.

So the current status is: **interesting candidate, no win claimed.** The repeatable-looking feature so far is reduced write activity, not superior loss.

## SOL-04 Arrow Monitor

Default synthetic 6-channel process with a hidden circulation reversal:

- first-half vs second-half skew-matrix cosine: -0.9907;
- first-half vs literal time reversal: -1.0000;
- sliding windows stay near +1 before the hidden reversal and become strongly negative afterward.

This validates the instrument on the toy process. Real usefulness requires testing on real multichannel streams where ordinary amplitude/spectral monitors miss an ordering change.

## Smoke harness

`python test_sol.py` -> `SOL smoke tests: PASS` on the development machine.

The repository rule remains: **keep the failures and ambiguous results beside the successes.**
