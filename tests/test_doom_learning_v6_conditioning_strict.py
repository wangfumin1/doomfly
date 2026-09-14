from doom_learning_v6.conditioning_strict_analysis import (
    CONDITIONS,
    build_schedule,
    evaluate_rows,
    protocol_integrity,
    schedule_metrics,
)


def row(plus, condition, selectivity=0.0, *, dose=800, duration=16000, unchanged=True, reset=True):
    before = [[10, 10], [10, 10]]
    after = before if unchanged else [[9, 10], [10, 10]]
    return {
        'plus': plus,
        'condition': condition,
        'selectivity': selectivity,
        'US_ms': 0 if condition == 'no_imposed_US' else dose,
        'training_ms': duration,
        'before_MBON': before,
        'after_MBON': after,
        'reset_restores': reset,
    }


def passing_rows():
    rows = []
    for plus in (0, 1):
        rows.extend([
            row(plus, 'paired', 0.30),
            row(plus, 'unpaired', 0.05),
            row(plus, 'frozen', 0.0),
            row(plus, 'no_imposed_US', 0.01),
        ])
    return rows


def test_protocol_is_duration_exposure_and_dose_matched():
    integrity = protocol_integrity()
    assert integrity['passed'] is True
    assert all(integrity['checks'].values())
    for plus in (0, 1):
        paired = schedule_metrics(build_schedule(plus, 'paired'))
        unpaired = schedule_metrics(build_schedule(plus, 'unpaired'))
        frozen = schedule_metrics(build_schedule(plus, 'frozen'))
        assert paired['training_ms'] == unpaired['training_ms'] == frozen['training_ms'] == 16000
        assert paired['cue_exposures'] == {0: 4, 1: 4}
        assert paired['US_ms'] == unpaired['US_ms'] == frozen['US_ms'] == 800
        assert paired['US_cue_overlap_trials'] == 4
        assert unpaired['US_cue_overlap_trials'] == 0


def test_strict_gate_passes_for_counterbalanced_pair_specific_effect():
    result = evaluate_rows(passing_rows())
    assert result['development_gate'] is True
    assert result['conditions'] == list(CONDITIONS)


def test_strict_gate_rejects_unpaired_equivalent_effect():
    rows = passing_rows()
    next(x for x in rows if x['plus'] == 0 and x['condition'] == 'unpaired')['selectivity'] = 0.30
    result = evaluate_rows(rows)
    assert result['development_gate'] is False
    assert result['checks']['cue_0_paired_beats_unpaired'] is False


def test_strict_gate_rejects_duration_or_dose_mismatch():
    rows = passing_rows()
    next(x for x in rows if x['plus'] == 0 and x['condition'] == 'unpaired')['training_ms'] = 15000
    next(x for x in rows if x['plus'] == 1 and x['condition'] == 'frozen')['US_ms'] = 600
    result = evaluate_rows(rows)
    assert result['development_gate'] is False
    assert result['checks']['cue_0_training_duration_matched'] is False
    assert result['checks']['cue_1_active_US_dose_matched'] is False
