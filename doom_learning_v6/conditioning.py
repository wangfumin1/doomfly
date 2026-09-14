"""Counterbalanced conditioning benchmark for the v6 full-connectome model.

This deliberately reuses the v5 assay timing so v5 and v6 can be compared
without changing both the neural model and the behavioral protocol at once.
It is a development benchmark, not evidence that a fly has learned Doom.
"""
import argparse
import time
from pathlib import Path

import numpy as np

from .brain import PARAMETERS
from .calibration import calibrated_brain
from doom_learning.common import OUT, capture_provenance, digest, save_json
from doom_learning_v2.vision import frame_for


CONDITIONS = ('paired', 'backward', 'frozen', 'no_imposed_US')


def evaluate_rows(rows):
    """Evaluate the counterbalanced development gate without running the brain.

    Keeping this pure makes the scientific decision rule cheap to unit-test and
    prevents analysis changes from being hidden inside an expensive full-graph run.
    """
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


def run(out, eta=.001, us_current=4.0):
    out = Path(out)
    if out.exists():
        raise ValueError('Fresh output path required')
    out.mkdir(parents=True)
    capture_provenance(out, additional=['doom_learning_v2', 'doom_learning_v6'])

    protocol = {
        'model': 'adaptive-centered-v6',
        'eta': eta,
        'parameters': PARAMETERS,
        'cue_names': ['left_blue', 'right_blue'],
        'conditions': list(CONDITIONS),
        'warmup_ms': 2000,
        'CS_ms': 1000,
        'US_current': us_current,
        'US_width_ms': 200,
        'forward_US_starts_ms': [200, 700, 1200, 1700],
        'backward_US_starts_ms': [0, 500, 1000, 1500],
        'backward_CS_start_ms': 2200,
        'post_training_dark_ms': 2000,
        'evaluation': (
            'Same 2 s warmup followed by 1 s cue and 0.4 s dark; '
            'all efficacies held fixed during probes.'
        ),
        'development_gate': (
            'Both cue counterbalances require positive paired selectivity, paired '
            'must exceed backward and no-US, paired/backward imposed-US dose must '
            'match, frozen weights must not change, and memory reset must restore '
            'pre-training responses.'
        ),
        'comparison_scope': (
            'Timing intentionally matches the v5 conditioning assay so a v5-v6 '
            'difference can be attributed to the model rather than a changed assay.'
        ),
        'status': (
            'Development assay only; not independent replication, not a survival '
            'experiment, and not evidence of learned game behavior.'
        ),
    }
    save_json(out / 'protocol.json', protocol)

    b = calibrated_brain(eta)
    save_json(out / 'calibration.json', b.calibration)
    frames = [frame_for('left_blue'), frame_for('right_blue')]
    dark = frame_for('black')

    def span(frame, milliseconds, learning=False, stim=None):
        total = np.zeros(b.n, dtype=np.int64)
        trace = []
        for _ in range(round(milliseconds / 10)):
            counts, _ = b.rgb_step(frame, 10, learning=learning, stimulation=stim)
            total += counts
            trace.append({
                'ms': b.sim_ms,
                'KC': int(counts[b.circuit['kc']].sum()),
                'DAN': counts[b.circuit['dan']].tolist(),
                'MBON': counts[b.circuit['mb']].tolist(),
            })
        return total, trace

    def warm():
        span(dark, 2000)

    def probe(frame):
        # Reset fast state while retaining the learned efficacy state, then freeze
        # it so the readout itself cannot continue training the model.
        b.reset(keep_memory=True)
        b.weights_frozen = True
        warm()
        cue_counts, trace = span(frame, 1000)
        tail_counts, tail = span(dark, 400)
        counts = cue_counts + tail_counts
        return {
            'MBON': counts[b.circuit['mb']].tolist(),
            'DAN': counts[b.circuit['dan']].tolist(),
            'KC': int(counts[b.circuit['kc']].sum()),
            'KC_pattern_sha256': digest(counts[b.circuit['kc']]),
            'trace': trace + tail,
        }

    rows = []
    for plus in (0, 1):
        for condition in CONDITIONS:
            b.reset()
            before = [probe(frame) for frame in frames]

            b.reset()
            b.weights_frozen = condition == 'frozen'
            warm()
            started = time.perf_counter()

            cs_start = 2200 if condition == 'backward' else 0
            cs_end = cs_start + 1000
            pulses = (
                protocol['backward_US_starts_ms']
                if condition == 'backward'
                else protocol['forward_US_starts_ms']
            )
            if condition == 'no_imposed_US':
                pulses = []

            duration = max(cs_end, 1900)
            training_counts = np.zeros(b.n, dtype=np.int64)
            training_trace = []
            delivered = 0
            for ms in range(0, duration + 2000, 10):
                us = any(start <= ms < start + 200 for start in pulses)
                delivered += 10 * int(us)
                counts, trace = span(
                    frames[plus] if cs_start <= ms < cs_end else dark,
                    10,
                    learning=condition != 'frozen',
                    stim=(b.circuit['dan'], us_current) if us else None,
                )
                training_counts += counts
                training_trace += trace

            memory = b.memory()
            after = [probe(frame) for frame in frames]
            suppression = [
                1 - sum(post['MBON']) / sum(pre['MBON']) if sum(pre['MBON']) else None
                for pre, post in zip(before, after)
            ]

            b.reset()
            erased = [probe(frame) for frame in frames]
            row = {
                'plus': plus,
                'condition': condition,
                'before': before,
                'after': after,
                'erased': erased,
                'suppression': suppression,
                'selectivity': (
                    suppression[plus] - suppression[1 - plus]
                    if all(value is not None for value in suppression)
                    else None
                ),
                'memory': memory,
                'training_KC': int(training_counts[b.circuit['kc']].sum()),
                'training_DAN': training_counts[b.circuit['dan']].tolist(),
                'US_ms': delivered,
                'training_trace': training_trace,
                'wall_seconds': time.perf_counter() - started,
            }
            save_json(out / f'{plus}-{condition}.json', row)

            slim = {
                key: value
                for key, value in row.items()
                if key not in ('before', 'after', 'erased', 'training_trace')
            }
            slim['before_MBON'] = [record['MBON'] for record in before]
            slim['after_MBON'] = [record['MBON'] for record in after]
            slim['reset_restores'] = all(
                pre['MBON'] == reset['MBON']
                and pre['KC_pattern_sha256'] == reset['KC_pattern_sha256']
                for pre, reset in zip(before, erased)
            )
            rows.append(slim)
            print(slim, flush=True)
            save_json(
                out / 'progress.json',
                {'completed': len(rows), 'total': 2 * len(CONDITIONS), 'latest': slim},
            )

    evaluation = evaluate_rows(rows)
    results = {
        'complete': True,
        **evaluation,
        'independently_validated': False,
        'survival_learning_demonstrated': False,
        'rows': rows,
    }
    save_json(out / 'results.json', results)
    print(results, flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(OUT / 'physiology-v6/conditioning-replication'))
    parser.add_argument('--eta', type=float, default=.001)
    parser.add_argument('--us-current', type=float, default=4.0)
    args = parser.parse_args()
    run(args.out, args.eta, args.us_current)
