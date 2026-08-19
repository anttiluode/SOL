"""SOL-02: Active Probe — choose the next ping that makes faults identifiable.

Rather than collecting arbitrary measurements, score candidate transient probes
by the information in their exact response Jacobian. This is the TWC
"negative capability / perturb the ambiguity" idea turned into an automatic
experiment selector for a KYY-style scatter fabric.
"""
from __future__ import annotations
import argparse
import numpy as np
from solcore import (
    greedy_probe_nodes, make_pulse_probe, stack_experiment,
    measure_experiment, gauss_newton_fit, information_score,
)


def recovery_rate(theta0, nodes, noise, trials=200, seed=0, ports=(0, 4)):
    rng = np.random.default_rng(seed)
    probes = [make_pulse_probe(theta0.size, n, steps=5) for n in nodes]
    hits = 0
    amp_err = []
    for _ in range(trials):
        edge = int(rng.integers(theta0.size))
        delta = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.025, 0.075))
        tt = theta0.copy(); tt[edge] += delta
        y = measure_experiment(tt, probes, ports=ports, noise_std=noise, rng=rng)
        fit = gauss_newton_fit(theta0, probes, y, ports=ports, iterations=5)
        d = fit - theta0
        guess = int(np.argmax(np.abs(d)))
        hits += int(guess == edge)
        amp_err.append(abs(d[edge] - delta))
    return hits / trials, float(np.median(amp_err))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, default=8e-4)
    ap.add_argument("--trials", type=int, default=160)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    n = 8; ports = (0, 4)
    rng = np.random.default_rng(args.seed)
    theta0 = rng.uniform(-0.6, 0.6, n)

    selected, scores = greedy_probe_nodes(theta0, n_select=4, ports=ports)
    repeated = [0, 0, 0, 0]
    random_nodes = list(rng.choice(n, size=4, replace=False))

    def score_nodes(nodes):
        probes = [make_pulse_probe(n, j, steps=5) for j in nodes]
        _, J = stack_experiment(theta0, probes, ports=ports)
        return information_score(J), np.linalg.matrix_rank(J, tol=1e-9), np.linalg.svd(J, compute_uv=False)

    print("=" * 74)
    print("SOL-02  ACTIVE PROBE — ask the system the most informative next question")
    print("=" * 74)
    print("greedy selection:")
    for k, (node, score) in enumerate(zip(selected, scores), 1):
        print(f"  ping {k}: node {node}   cumulative logdet score {score:+.3f}")

    print("\nmeasurement-set geometry:")
    for name, nodes in [("active", selected), ("random", random_nodes), ("repeat-0", repeated)]:
        sc, rank, sv = score_nodes(nodes)
        nz = sv[sv > 1e-9]
        cond = (nz.max() / nz.min()) if nz.size else np.inf
        print(f"  {name:8s} nodes={nodes} rank={rank}/8 score={sc:+.2f} observable-cond={cond:.2e}")

    print("\nfault-localization Monte Carlo:")
    for name, nodes in [("active", selected), ("random", random_nodes), ("repeat-0", repeated)]:
        hit, err = recovery_rate(theta0, nodes, args.noise, trials=args.trials,
                                 seed=args.seed + 100, ports=ports)
        print(f"  {name:8s}: top-1 {hit*100:5.1f}%   median true-edge amplitude error {err:.5f} rad")

    print("\nInterpretation: if a measurement cannot distinguish internal changes, do not")
    print("optimize harder. Change the experiment. Here the compiler itself proposes the")
    print("next transient injection that makes the local geometry more observable.")


if __name__ == "__main__":
    main()
