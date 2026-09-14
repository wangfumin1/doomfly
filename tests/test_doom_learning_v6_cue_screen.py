import pytest

from doom_learning_v6.cue_screen_analysis import analyze_cues


def record(label, kc, pattern, mbon=(10, 10)):
    return {
        'label': label,
        'KC': kc,
        'KC_pattern_sha256': pattern,
        'MBON': list(mbon),
    }


def test_recommends_balanced_distinguishable_pair():
    rows = [
        record('black', 100, 'dark'),
        record('left_blue', 1000, 'left'),
        record('right_blue', 1200, 'right'),
        record('vertical', 5000, 'vertical'),
    ]
    result = analyze_cues(rows, maximum_activity_ratio=2.0)
    assert result['legacy_pair_ready'] is True
    assert result['recommended_pair']['cue_a'] == 'left_blue'
    assert result['recommended_pair']['cue_b'] == 'right_blue'
    assert result['recommended_pair']['conditioning_ready'] is True


def test_rejects_extreme_activity_imbalance():
    rows = [
        record('black', 100, 'dark'),
        record('left_blue', 925000, 'left'),
        record('right_blue', 172, 'right'),
    ]
    result = analyze_cues(rows, maximum_activity_ratio=2.0)
    assert result['legacy_pair_ready'] is False
    assert result['legacy_pair']['activity_balanced'] is False
    assert result['ready_pair_count'] == 0


def test_rejects_cue_that_is_identical_to_dark():
    rows = [
        record('black', 100, 'dark'),
        record('left_blue', 100, 'dark'),
        record('right_blue', 110, 'right'),
    ]
    result = analyze_cues(rows)
    assert result['legacy_pair_ready'] is False
    assert result['legacy_pair']['cue_a_differs_from_dark'] is False


def test_rejects_missing_mbon_readout():
    rows = [
        record('black', 100, 'dark'),
        record('left_blue', 1000, 'left', (0, 0)),
        record('right_blue', 1000, 'right'),
    ]
    result = analyze_cues(rows)
    assert result['legacy_pair']['MBON_readout_available'] is False
    assert result['legacy_pair_ready'] is False


def test_black_control_is_required():
    with pytest.raises(ValueError, match='black control'):
        analyze_cues([record('left_blue', 100, 'left')])
