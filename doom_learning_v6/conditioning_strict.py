"""Preflight-gated, timing-matched differential conditioning for v6.

Run ``python -m doom_learning_v6.cue_screen`` first. This experiment refuses to
train on a cue pair that did not pass the visual preflight. Both cues receive the
same number and duration of presentations in every condition. Paired, unpaired
and frozen controls receive the same imposed dopamine dose; only its contingency
with the CS+ differs.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .brain import PARAMETERS
from .calibration import calibrated_brain
from .conditioning_strict_analysis import (
    CONDITIONS,
    build_schedule,
    evaluate_rows,
    protocol_integrity,
    schedule_metrics,
)
from doom_learning.common import OUT, capture_provenance, digest, save_json
from doom_learning_v2.vision import frame_for


def _select_cues(preflight_path, cue_a=None, cue_b=None):
    path = Path(preflight_path)
    payload = json.loads(path.read_text())
    analysis = payload['analysis']
    if cue_a is None and cue_b is None:
        pair = analysis.get('recommended_pair')
        if not pair or not pair.get('conditioning_ready'):
            raise ValueError('Cue preflight has no conditioning-ready pair')
        cue_a, cue_b = pair['cue_a'], pair['cue_b']
    elif not cue_a or not cue_b:
        raise ValueError('Specify both cue_a and cue_b, or neither')
    else:
        pair = next((
            item for item in analysis.get('pairs', [])
            if {item['cue_a'], item['cue_b']} == {cue_a, cue_b}
        ), None)
        if not pair or not pair.get('conditioning_ready'):
            raise ValueError(f'Selected cue pair failed preflight: {cue_a}, {cue_b}')
    return (cue_a, cue_b), pair, hashlib.sha256(path.read_bytes()).hexdigest()


def run(out, preflight, *, eta=.001, us_current=4.0, cue_a=None, cue_b=None):
    out = Path(out)
    if out.exists():
        raise ValueError('Fresh output path required')
    cues, pair_metrics, preflight_sha256 = _select_cues(preflight, cue_a, cue_b)
    integrity = protocol_integrity()
    if not integrity['passed']:
        raise RuntimeError('Strict conditioning protocol failed its own integrity checks')

    out.mkdir(parents=True)
    capture_provenance(out, additional=['doom_learning_v2', 'doom_learning_v6'])
    protocol = {
        'model': 'adaptive-centered-v6',
        'eta': eta,
        'parameters': PARAMETERS,
        'cue_names': list(cues),
        'conditions': list(CONDITIONS),
        'US_current': us_current,
        'protocol_integrity': integrity,
        'selected_pair_preflight': pair_metrics,
        'preflight_sha256': preflight_sha256,
        'training_design': (
            'Eight 2 s trials per condition; each cue appears four times. Cue is '
            'shown from 0.5-1.5 s. Paired/frozen US is a 200 ms PPL101 stimulus '
            'inside every CS+ trial. Unpaired receives four equal-dose pulses only '
            'outside cue windows, split across both cue identities. no-US receives '
            'the same visual exposure without imposed dopamine.'
        ),
        'evaluation': (
            'Each probe resets fast neural state while retaining learned efficacy, '
            'warms in darkness for 2 s, then presents 1 s cue plus 0.4 s dark tail. '
            'Weights are frozen during every probe.'
        ),
        'status': (
            'Development assay only. A passing gate is evidence for a cue-specific '
            'effect under this model and protocol, not independent biological '
            'validation and not proof of learned Doom survival.'
        ),
    }
    save_json(out / 'protocol.json', protocol)

    b = calibrated_brain(eta)
    save_json(out / 'calibration.json', b.calibration)
    frames = [frame_for(cues[0]), frame_for(cues[1])]
    dark = frame_for('black')

    def span(frame, milliseconds, *, learning=False, stim=None):
        total = np.zeros(b.n, dtype=np.int64)
        for _ in range(round(milliseconds / 10)):
            counts, _ = b.rgb_step(frame, 10, learning=learning, stimulation=stim)
            total += counts
        return total

    def warm():
        span(dark, 2000)

    def probe(frame):
        b.reset(keep_memory=True)
        b.weights_frozen = True
        warm()
        counts = span(frame, 1000) + span(dark, 400)
        return {
            'MBON': counts[b.circuit['mb']].tolist(),
            'DAN': counts[b.circuit['dan']].tolist(),
            'KC': int(counts[b.circuit['kc']].sum()),
            'active_KC': int(np.count_nonzero(counts[b.circuit['kc']])),
            'KC_pattern_sha256': digest(counts[b.circuit['kc']]),
        }

    rows = []
    for plus in (0, 1):
        for condition in CONDITIONS:
            b.reset()
            before = [probe(frame) for frame in frames]

            b.reset()
            b.weights_frozen = condition == 'frozen'
            warm()
            schedule = build_schedule(plus, condition)
            schedule_info = schedule_metrics(schedule)
            started = time.perf_counter()
            training_counts = np.zeros(b.n, dtype=np.int64)
            training_trials = []
            delivered = 0

            for trial in schedule:
                trial_counts = np.zeros(b.n, dtype=np.int64)
                trial_us_ms = 0
                for ms in range(0, trial['trial_ms'], 10):
                    cue_on = trial['cue_start_ms'] <= ms < trial['cue_end_ms']
                    us_on = (
                        trial['us_start_ms'] is not None
                        and trial['us_start_ms'] <= ms < trial['us_end_ms']
                    )
                    counts = span(
                        frames[trial['cue']] if cue_on else dark,
                        10,
                        learning=condition != 'frozen',
                        stim=(b.circuit['dan'], us_current) if us_on else None,
                    )
                    trial_counts += counts
                    trial_us_ms += 10 * int(us_on)
                delivered += trial_us_ms
                training_counts += trial_counts
                training_trials.append({
                    **trial,
                    'US_ms': trial_us_ms,
                    'KC': int(trial_counts[b.circuit['kc']].sum()),
                    'DAN': trial_counts[b.circuit['dan']].tolist(),
                    'MBON': trial_counts[b.circuit['mb']].tolist(),
                })

            if delivered != schedule_info['US_ms']:
                raise RuntimeError('Delivered US dose differs from pure protocol schedule')

            memory = b.memory()
            after = [probe(frame) for frame in frames]
            suppression = [
                1 - sum(post['MBON']) / sum(pre['MBON']) if sum(pre['MBON']) else None
                for pre, post in zip(before, after)
            ]

            b.reset()
            erased = [probe(frame) for frame in frames]
            reset_restores = all(
                pre['MBON'] == reset['MBON']
                and pre['KC_pattern_sha256'] == reset['KC_pattern_sha256']
                for pre, reset in zip(before, erased)
            )
            row = {
                'plus': plus,
                'condition': condition,
                'cue_names': list(cues),
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
                'training_ms': schedule_info['training_ms'],
                'training_trials': training_trials,
                'wall_seconds': time.perf_counter() - started,
                'reset_restores': reset_restores,
            }
            save_json(out / f'{plus}-{condition}.json', row)

            slim = {
                key: value for key, value in row.items()
                if key not in ('before', 'after', 'erased', 'training_trials')
            }
            slim['before_MBON'] = [record['MBON'] for record in before]
            slim['after_MBON'] = [record['MBON'] for record in after]
            rows.append(slim)
            print(slim, flush=True)
            save_json(
                out / 'progress.json',
                {'completed': len(rows), 'total': 2 * len(CONDITIONS), 'latest': slim},
            )

    evaluation = evaluate_rows(rows)
    results = {
        'complete': True,
        'cue_names': list(cues),
        'preflight_sha256': preflight_sha256,
        **evaluation,
        'independently_validated': False,
        'survival_learning_demonstrated': False,
        'rows': rows,
    }
    save_json(out / 'results.json', results)
    print({key: value for key, value in results.items() if key != 'rows'}, flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(OUT / 'physiology-v6/conditioning-strict'))
    parser.add_argument('--preflight', default=str(OUT / 'physiology-v6/cue-screen/results.json'))
    parser.add_argument('--eta', type=float, default=.001)
    parser.add_argument('--us-current', type=float, default=4.0)
    parser.add_argument('--cue-a')
    parser.add_argument('--cue-b')
    args = parser.parse_args()
    run(
        args.out,
        args.preflight,
        eta=args.eta,
        us_current=args.us_current,
        cue_a=args.cue_a,
        cue_b=args.cue_b,
    )
