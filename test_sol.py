import numpy as np
from solcore import make_pulse_probe, run_probe, finite_difference_jacobian, greedy_probe_nodes
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path


def load_numbered(name: str, module_name: str):
    path = Path(__file__).with_name(name)
    spec = spec_from_file_location(module_name, path)
    m = module_from_spec(spec); spec.loader.exec_module(m)
    return m


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
    m = load_numbered("04_arrow_monitor.py", "arrow_monitor")
    X = m.simulate(steps=2000, flip_at=2000, seed=3)
    A = m.lag_skew(X, 1)
    Ar = m.lag_skew(X[::-1], 1)
    assert m.cosine_matrix(A, Ar) < -0.95


def test_zombie_memory_audit():
    m = load_numbered("05_zombie_memory_audit.py", "zombie_memory")
    fake, true = m.demo(max_depth=4, threshold=0.5)
    assert fake.found
    assert fake.output_gap > 1.5
    assert fake.hidden_gap > 1.5
    assert not true.found
    assert true.hidden_gap < 1e-10


def test_behavior_gauge_doctor():
    m = load_numbered("06_behavior_gauge_doctor.py", "behavior_gauge")
    r = m.demo()
    assert r["relation_rank"] == 1
    assert r["behavior_rank"] == 2
    assert r["relation_repair_defect"] < 1e-8
    assert r["relation_repair_port_error"] > 1e-3
    assert r["full_repair_defect"] < 1e-8
    assert r["common_relation_defect"] < 1e-8
    assert r["common_behavior_defect"] > 1e-3


def test_phase_handoff_compiler():
    m = load_numbered("07_phase_handoff_compiler.py", "phase_handoff")
    alpha, margin = m.compile_alpha(samples=1001)
    assert abs(alpha - np.pi / 8) < 2e-3
    assert margin > 0.12 * np.pi
    good = m.simulate(alpha, sigma=0.08, trials=10000, seed=2)
    bad = m.simulate(np.pi / 4, sigma=0.08, trials=10000, seed=2)
    assert good > 0.98
    assert 0.45 < bad < 0.55


if __name__ == "__main__":
    test_exact_jacobian()
    test_active_probe_unique()
    test_arrow_time_reverse()
    test_zombie_memory_audit()
    test_behavior_gauge_doctor()
    test_phase_handoff_compiler()
    print("SOL smoke tests: PASS")
