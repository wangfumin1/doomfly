import numpy as np

from doom_learning_v6.stimuli import frame_for, pixel_energy


def test_checker_phases_are_energy_matched_but_spatially_distinct():
    a = frame_for('checker_a')
    b = frame_for('checker_b')
    assert pixel_energy('checker_a') == pixel_energy('checker_b')
    assert not np.array_equal(a, b)
    assert np.count_nonzero(a) == np.count_nonzero(b)


def test_quadrant_phases_are_energy_matched_but_spatially_distinct():
    a = frame_for('quadrants_a')
    b = frame_for('quadrants_b')
    assert pixel_energy('quadrants_a') == pixel_energy('quadrants_b')
    assert not np.array_equal(a, b)
    assert np.count_nonzero(a) == np.count_nonzero(b)


def test_research_stimulus_wrapper_preserves_legacy_black():
    black = frame_for('black')
    assert black.shape == (240, 320, 3)
    assert black.dtype == np.uint8
    assert not black.any()
