import numpy as np
from solcore import make_pulse_probe, run_probe, finite_difference_jacobian, greedy_probe_nodes
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path


def test_exact_jacobian():
    rng = np.random.default_rng(0)
    theta = rng.uniform(-0.5, 0.5, 8)
    p = make_pulse_probe(8, 3, steps=4)
    _, J = run_probe(theta, p, ports=(0, 4))
    Jfd = finite_difference_jacobian(theta, p, ports=(0, 4))
    assert np.max(np.abs(J - Jfd)) < 2e-5


def test_active_probe_unique():
    theta = np.linspace(-0.4, 0.4, 8)
    nodes, scores = greedy_probe_nodes(theta, 4)
    assert len(nodes) == len(set(nodes)) == 4
    assert all(np.isfinite(scores))


def test_arrow_time_reverse():
    path = Path(__file__).with_name("04_arrow_monitor.py")
    spec = spec_from_file_location("arrow_monitor", path)
    m = module_from_spec(spec); spec.loader.exec_module(m)
    X = m.simulate(steps=2000, flip_at=2000, seed=3)
    A = m.lag_skew(X, 1)
    Ar = m.lag_skew(X[::-1], 1)
    assert m.cosine_matrix(A, Ar) < -0.95


if __name__ == "__main__":
    test_exact_jacobian(); test_active_probe_unique(); test_arrow_time_reverse()
    print("SOL smoke tests: PASS")
