from doom_learning_v6.conditioning_analysis import CONDITIONS, evaluate_rows


def row(plus, condition, selectivity=0.0, *, dose=800, unchanged=True, reset=True):
    before = [[10, 10], [10, 10]]
    after = before if unchanged else [[9, 10], [10, 10]]
    return {
        'plus': plus,
        'condition': condition,
        'selectivity': selectivity,
        'US_ms': 0 if condition == 'no_imposed_US' else dose,
        'before_MBON': before,
        'after_MBON': after,
        'reset_restores': reset,
    }


def passing_rows():
    rows = []
    for plus in (0, 1):
        rows.extend([
            row(plus, 'paired', 0.25),
            row(plus, 'backward', 0.05),
            row(plus, 'frozen', 0.0),
            row(plus, 'no_imposed_US', 0.01),
        ])
    return rows


def test_counterbalanced_gate_passes_only_when_all_controls_pass():
    result = evaluate_rows(passing_rows())
    assert result['development_gate'] is True
    assert result['conditions'] == list(CONDITIONS)
    assert all(result['checks'].values())


def test_gate_rejects_nonselective_paired_conditioning():
    rows = passing_rows()
    next(x for x in rows if x['plus'] == 1 and x['condition'] == 'paired')['selectivity'] = -0.01
    result = evaluate_rows(rows)
    assert result['development_gate'] is False
    assert result['checks']['cue_1_paired_positive'] is False


def test_gate_rejects_dose_mismatch_and_failed_reset():
    rows = passing_rows()
    next(x for x in rows if x['plus'] == 0 and x['condition'] == 'backward')['US_ms'] = 600
    next(x for x in rows if x['plus'] == 1 and x['condition'] == 'paired')['reset_restores'] = False
    result = evaluate_rows(rows)
    assert result['development_gate'] is False
    assert result['checks']['cue_0_paired_backward_dose_matched'] is False
    assert result['checks']['cue_1_memory_reset_restores'] is False


def test_gate_rejects_frozen_weight_readout_change():
    rows = passing_rows()
    frozen = next(x for x in rows if x['plus'] == 0 and x['condition'] == 'frozen')
    frozen['after_MBON'] = [[9, 10], [10, 10]]
    result = evaluate_rows(rows)
    assert result['development_gate'] is False
    assert result['checks']['cue_0_frozen_unchanged'] is False


def test_gate_rejects_missing_condition():
    rows = [
        x for x in passing_rows()
        if not (x['plus'] == 0 and x['condition'] == 'backward')
    ]
    result = evaluate_rows(rows)
    assert result['development_gate'] is False
    assert result['checks']['cue_0_complete'] is False
