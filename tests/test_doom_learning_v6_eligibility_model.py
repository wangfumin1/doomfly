"""Prepared-runtime checks for the eligibility-LTD v6 comparison model.

These tests require the native neural kernel and are intentionally excluded from
the lightweight GitHub Actions job.
"""
import numpy as np

from doom_learning_v6.brain import MemoryBrain as CenteredBrain
from doom_learning_v6.eligibility_model import EligibilityMemoryBrain


def fixture_brain(tmp_path, cls, eta=.001):
    path = tmp_path / 'graph.npz'
    n = 6
    np.savez(
        path,
        ptr=np.array([0, 1, 1, 2, 2, 3, 4], dtype=np.int64),
        post=np.array([1, 3, 1, 3], dtype=np.int32),
        weight=np.array([20, 20, .275, .275], dtype=np.float32),
        ids=np.arange(n, dtype=np.int64),
        retina=np.empty(0, dtype=np.int32),
        uv=np.empty((0, 2), dtype=np.float32),
        lamina=np.empty(0, dtype=np.int32),
        sugar=np.empty(0, dtype=np.int32),
        superclass=np.array(['test'] * n),
    )
    circuit = {
        'edges': np.array([0, 1], dtype=np.int64),
        'pre': np.array([0, 2], dtype=np.int32),
        'kc_mask': np.array([1, 0, 1, 0, 0, 0], dtype=np.uint8),
        'dan_index': np.array([-1, -1, -1, -1, 0, 1], dtype=np.int8),
        'gain': np.array([[1, 0], [0, 1]], dtype=np.float32),
        'kc': np.array([0, 2]),
        'mb': np.array([1, 3]),
        'dan': np.array([4, 5]),
    }
    return cls(
        path,
        eta=eta,
        circuit=circuit,
        modulation_mask=circuit['dan_index'] >= 0,
        kc_rest=-52.0,
        adaptation_jump=0.0,
    )


def test_no_learning_matches_centered_v6_neural_dynamics(tmp_path):
    candidate = fixture_brain(tmp_path, EligibilityMemoryBrain)
    centered = fixture_brain(tmp_path, CenteredBrain)
    for stimulus in [([0], 20), ([4], 20), ([0, 4], 20)]:
        a, _ = candidate.step([], 100, learning=False, stimulation=stimulus, lamina_bias=0)
        b, _ = centered.step([], 100, learning=False, stimulation=stimulus, lamina_bias=0)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(candidate.v, centered.v, atol=.002, rtol=0)
        np.testing.assert_array_equal(candidate.weight, centered.weight)


def test_kc_dan_pairing_depresses_only_supported_edge(tmp_path):
    brain = fixture_brain(tmp_path, EligibilityMemoryBrain)
    before = brain.weight.copy()
    brain.step([], 100, learning=True, stimulation=([0, 4], 30), lamina_bias=0)
    assert brain.weight[0] < before[0]
    np.testing.assert_array_equal(brain.weight[1:], before[1:])
    assert brain.memory()['model'] == 'eligibility-ltd-v6'


def test_no_dan_and_frozen_mode_do_not_change_weights(tmp_path):
    brain = fixture_brain(tmp_path, EligibilityMemoryBrain)
    before = brain.weight.copy()
    brain.step([], 100, learning=True, stimulation=([0], 30), lamina_bias=0)
    np.testing.assert_array_equal(brain.weight, before)

    brain.weights_frozen = True
    brain.step([], 100, learning=True, stimulation=([0, 4], 30), lamina_bias=0)
    np.testing.assert_array_equal(brain.weight, before)


def test_reset_keeps_or_erases_candidate_memory_explicitly(tmp_path):
    brain = fixture_brain(tmp_path, EligibilityMemoryBrain)
    brain.step([], 100, learning=True, stimulation=([0, 4], 30), lamina_bias=0)
    learned = brain.weight.copy()
    assert brain.memory()['changed_edges'] == 1

    brain.reset(keep_memory=True)
    np.testing.assert_array_equal(brain.weight, learned)
    assert not brain.eligibility.any()

    brain.reset()
    np.testing.assert_array_equal(brain.weight[:2], brain.baseline_plastic)
