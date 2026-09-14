"""Screen visual cues before attempting v6 associative conditioning.

The conditioning assay is not interpretable if one cue barely drives Kenyon cells
while the other drives the mushroom body strongly. This preflight measures each
candidate after the same dark warmup and reports balanced, distinguishable pairs.
"""
import argparse
from pathlib import Path

import numpy as np

from .calibration import calibrated_brain
from .cue_screen_analysis import DEFAULT_CUES, analyze_cues
from .stimuli import frame_for, pixel_energy
from doom_learning.common import OUT, capture_provenance, digest, save_json


def run(out, *, maximum_activity_ratio=2.0):
    out = Path(out)
    if out.exists():
        raise ValueError('Fresh output path required')
    out.mkdir(parents=True)
    capture_provenance(out, additional=['doom_learning_v2', 'doom_learning_v6'])

    b = calibrated_brain(eta=0.0)
    dark = frame_for('black')
    labels = ('black', *DEFAULT_CUES)

    def span(frame, milliseconds):
        total = np.zeros(b.n, dtype=np.int64)
        for _ in range(round(milliseconds / 10)):
            counts, _ = b.rgb_step(frame, 10, learning=False)
            total += counts
        return total

    rows = []
    for label in labels:
        b.reset()
        b.weights_frozen = True
        span(dark, 2000)
        cue = frame_for(label)
        counts = span(cue, 1000) + span(dark, 400)
        row = {
            'label': label,
            'KC': int(counts[b.circuit['kc']].sum()),
            'active_KC': int(np.count_nonzero(counts[b.circuit['kc']])),
            'MBON': counts[b.circuit['mb']].tolist(),
            'DAN': counts[b.circuit['dan']].tolist(),
            'R1_R6': int(counts[b.retina].sum()),
            'R8': int(counts[b.r8].sum()),
            'KC_pattern_sha256': digest(counts[b.circuit['kc']]),
            'input_sha256': digest(cue),
            'pixel_energy': pixel_energy(label),
        }
        rows.append(row)
        save_json(out / f'{label}.json', row)
        print(row, flush=True)

    analysis = analyze_cues(rows, maximum_activity_ratio=maximum_activity_ratio)
    result = {
        'model': 'adaptive-centered-v6',
        'learning_enabled': False,
        'warmup_ms': 2000,
        'cue_ms': 1000,
        'tail_dark_ms': 400,
        'rows': rows,
        'analysis': analysis,
        'conditioning_preflight_passed': analysis['ready_pair_count'] > 0,
        'status': (
            'Visual preflight only. Passing means at least one candidate pair is '
            'activity-balanced and distinguishable at KC/MBON readout; it does not '
            'establish associative learning or biological visual validity. The new '
            'checker/quadrant cue families are pixel-energy matched by construction, '
            'but still require neural activity balance here.'
        ),
    }
    save_json(out / 'results.json', result)
    print(result['analysis'], flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(OUT / 'physiology-v6/cue-screen'))
    parser.add_argument('--maximum-activity-ratio', type=float, default=2.0)
    args = parser.parse_args()
    run(args.out, maximum_activity_ratio=args.maximum_activity_ratio)
