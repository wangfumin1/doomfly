"""Pure decision logic for the v6 counterbalanced conditioning benchmark.

This module intentionally imports no simulator or numerical dependencies so the
scientific gate can be tested in lightweight CI without a built neural kernel or
MaleCNS dataset.
"""

CONDITIONS = ('paired', 'backward', 'frozen', 'no_imposed_US')


def evaluate_rows(rows):
    checks = {}
    for plus in (0, 1):
        subset = {row['condition']: row for row in rows if row['plus'] == plus}
        missing = sorted(set(CONDITIONS) - set(subset))
        if missing:
            checks[f'cue_{plus}_complete'] = False
            continue

        paired = subset['paired']['selectivity']
        backward = subset['backward']['selectivity']
        no_us = subset['no_imposed_US']['selectivity']
        frozen = subset['frozen']

        checks[f'cue_{plus}_complete'] = True
        checks[f'cue_{plus}_paired_positive'] = paired is not None and paired > 0
        checks[f'cue_{plus}_paired_beats_backward'] = (
            paired is not None and backward is not None and paired > backward
        )
        checks[f'cue_{plus}_paired_beats_no_US'] = (
            paired is not None and no_us is not None and paired > no_us
        )
        checks[f'cue_{plus}_frozen_unchanged'] = (
            frozen['before_MBON'] == frozen['after_MBON']
        )
        checks[f'cue_{plus}_memory_reset_restores'] = all(
            row.get('reset_restores') is True for row in subset.values()
        )
        checks[f'cue_{plus}_paired_backward_dose_matched'] = (
            subset['paired']['US_ms'] == subset['backward']['US_ms']
        )
        checks[f'cue_{plus}_no_US_zero_dose'] = subset['no_imposed_US']['US_ms'] == 0

    return {
        'checks': checks,
        'development_gate': bool(checks) and all(checks.values()),
        'counterbalances': 2,
        'conditions': list(CONDITIONS),
    }
