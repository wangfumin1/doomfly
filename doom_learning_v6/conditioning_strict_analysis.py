"""Pure protocol and decision logic for strict v6 differential conditioning."""

CONDITIONS = ('paired', 'unpaired', 'frozen', 'no_imposed_US')
CUE_SEQUENCE = (0, 1, 1, 0, 1, 0, 0, 1)
TRIAL_MS = 2000
CUE_START_MS = 500
CUE_END_MS = 1500
US_WIDTH_MS = 200
PAIRED_US_START_MS = 1100
UNPAIRED_US = {0: 100, 1: 1700, 4: 100, 5: 1700}


def build_schedule(plus, condition):
    if plus not in (0, 1):
        raise ValueError('plus must be cue index 0 or 1')
    if condition not in CONDITIONS:
        raise ValueError(f'unknown condition: {condition}')
    schedule = []
    for trial, cue in enumerate(CUE_SEQUENCE):
        us_start = None
        if condition in ('paired', 'frozen') and cue == plus:
            us_start = PAIRED_US_START_MS
        elif condition == 'unpaired':
            us_start = UNPAIRED_US.get(trial)
        schedule.append({
            'trial': trial,
            'cue': cue,
            'cue_start_ms': CUE_START_MS,
            'cue_end_ms': CUE_END_MS,
            'us_start_ms': us_start,
            'us_end_ms': None if us_start is None else us_start + US_WIDTH_MS,
            'trial_ms': TRIAL_MS,
        })
    return schedule


def schedule_metrics(schedule):
    cue_exposures = {
        0: sum(row['cue'] == 0 for row in schedule),
        1: sum(row['cue'] == 1 for row in schedule),
    }
    us_pulses = sum(row['us_start_ms'] is not None for row in schedule)
    us_ms = us_pulses * US_WIDTH_MS
    overlaps = 0
    for row in schedule:
        if row['us_start_ms'] is None:
            continue
        if row['us_start_ms'] < row['cue_end_ms'] and row['us_end_ms'] > row['cue_start_ms']:
            overlaps += 1
    return {
        'training_ms': sum(row['trial_ms'] for row in schedule),
        'cue_exposures': cue_exposures,
        'us_pulses': us_pulses,
        'US_ms': us_ms,
        'US_cue_overlap_trials': overlaps,
    }


def protocol_integrity():
    metrics = {
        plus: {condition: schedule_metrics(build_schedule(plus, condition))
               for condition in CONDITIONS}
        for plus in (0, 1)
    }
    checks = {}
    for plus in (0, 1):
        current = metrics[plus]
        durations = {x['training_ms'] for x in current.values()}
        exposures = {tuple(sorted(x['cue_exposures'].items())) for x in current.values()}
        checks[f'cue_{plus}_duration_matched'] = len(durations) == 1
        checks[f'cue_{plus}_cue_exposure_matched'] = len(exposures) == 1
        checks[f'cue_{plus}_active_US_dose_matched'] = (
            current['paired']['US_ms'] == current['unpaired']['US_ms']
            == current['frozen']['US_ms']
        )
        checks[f'cue_{plus}_no_US_zero_dose'] = current['no_imposed_US']['US_ms'] == 0
        checks[f'cue_{plus}_paired_US_overlaps_only_plus_trials'] = (
            current['paired']['US_cue_overlap_trials'] == 4
        )
        checks[f'cue_{plus}_unpaired_US_never_overlaps_cue'] = (
            current['unpaired']['US_cue_overlap_trials'] == 0
        )
    return {'checks': checks, 'passed': all(checks.values()), 'metrics': metrics}


def evaluate_rows(rows):
    checks = {}
    for plus in (0, 1):
        subset = {row['condition']: row for row in rows if row['plus'] == plus}
        if set(subset) != set(CONDITIONS):
            checks[f'cue_{plus}_complete'] = False
            continue
        paired = subset['paired']['selectivity']
        unpaired = subset['unpaired']['selectivity']
        no_us = subset['no_imposed_US']['selectivity']
        frozen = subset['frozen']
        checks[f'cue_{plus}_complete'] = True
        checks[f'cue_{plus}_paired_positive'] = paired is not None and paired > 0
        checks[f'cue_{plus}_paired_beats_unpaired'] = (
            paired is not None and unpaired is not None and paired > unpaired
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
        checks[f'cue_{plus}_active_US_dose_matched'] = (
            subset['paired']['US_ms'] == subset['unpaired']['US_ms']
            == subset['frozen']['US_ms']
        )
        checks[f'cue_{plus}_no_US_zero_dose'] = subset['no_imposed_US']['US_ms'] == 0
        checks[f'cue_{plus}_training_duration_matched'] = len({
            row['training_ms'] for row in subset.values()
        }) == 1

    integrity = protocol_integrity()
    return {
        'checks': checks,
        'protocol_integrity': integrity,
        'development_gate': (
            bool(checks) and all(checks.values()) and integrity['passed']
        ),
        'counterbalances': 2,
        'conditions': list(CONDITIONS),
    }
