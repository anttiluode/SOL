"""SOL-01: Edge Doctor — self-diagnosis for a local scatter network.

Simulates one drifting physical/local edge, observes only two output ports under
several transient probes, and uses exact forward sensitivities to locate it.

This is TWC-style diagnosis applied to KYY-style local reciprocal recurrence.
"""
from __future__ import annotations
import argparse
import numpy as np
from solcore import (
    make_pulse_probe, measure_experiment, diagnose_linearized,
    gauss_newton_fit, stack_experiment, finite_difference_jacobian,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=int, default=8)
    ap.add_argument("--fault-edge", type=int, default=5)
    ap.add_argument("--fault", type=float, default=0.06, help="hidden edge-angle drift in radians")
    ap.add_argument("--noise", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=4)
    args = ap.parse_args()

    if args.state != 8:
        ports = (0, args.state // 2)
    else:
        ports = (0, 4)
    rng = np.random.default_rng(args.seed)
    theta0 = rng.uniform(-0.55, 0.55, args.state)
    fault_edge = args.fault_edge % args.state
    theta_true = theta0.copy(); theta_true[fault_edge] += args.fault

    # Four independent pings; only two values are observed from each ping.
    nodes = [0, 1, 3, 6] if args.state == 8 else list(range(min(4, args.state)))
    probes = [make_pulse_probe(args.state, n, steps=5) for n in nodes]
    measured = measure_experiment(theta_true, probes, ports=ports, noise_std=args.noise, rng=rng)

    diag = diagnose_linearized(theta0, probes, measured, ports=ports)
    theta_fit = gauss_newton_fit(theta0, probes, measured, ports=ports, iterations=6)
    dfit = theta_fit - theta0

    # Exact-Jacobian sanity check on one probe.
    _, J_exact = stack_experiment(theta0, [probes[0]], ports=ports)
    J_fd = finite_difference_jacobian(theta0, probes[0], ports=ports)
    jac_err = float(np.max(np.abs(J_exact - J_fd)))

    print("=" * 74)
    print("SOL-01  EDGE DOCTOR — diagnose a drifting local scatter edge")
    print("=" * 74)
    print(f"state/edges       : {args.state}")
    print(f"observed ports    : {ports}")
    print(f"probe nodes       : {nodes}")
    print(f"hidden fault      : edge {fault_edge}, {args.fault:+.5f} rad")
    print(f"measurement noise : sigma={args.noise:g}")
    print(f"Jacobian rank     : {diag.jacobian_rank}/{args.state}")
    print(f"Jacobian condition: {diag.condition_number:.2e}")
    print(f"exact-vs-FD maxerr: {jac_err:.3e}")
    print("\nlinearized top-4 edge estimates:")
    for e in diag.ranking[:4]:
        print(f"  edge {e}: {diag.delta_hat[e]:+.6f} rad")
    order_fit = np.argsort(-np.abs(dfit))
    print("\nGauss-Newton top-4 edge estimates:")
    for e in order_fit[:4]:
        print(f"  edge {e}: {dfit[e]:+.6f} rad")
    print(f"\nRESULT: top-1 edge = {order_fit[0]}  true edge = {fault_edge}  "
          f"{'PASS' if order_fit[0] == fault_edge else 'FAIL'}")
    print("\nWhy this matters: a geometry-constrained recurrent/physical network exposes")
    print("named local parameters. The same structure used to compute can also tell you")
    print("which local element drifted, instead of treating the whole network as a blob.")


if __name__ == "__main__":
    main()
