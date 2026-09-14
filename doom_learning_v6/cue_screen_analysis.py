"""Pure analysis for pre-conditioning visual cue screening."""
import math


DEFAULT_CUES = (
    'blue', 'green', 'white', 'left_blue', 'right_blue', 'vertical', 'horizontal',
    'checker_a', 'checker_b', 'quadrants_a', 'quadrants_b',
)
LEGACY_PAIR = ('left_blue', 'right_blue')


def _pair_metrics(a, b, dark, maximum_activity_ratio):
    ka = int(a['KC'])
    kb = int(b['KC'])
    minimum = min(ka, kb)
    ratio = math.inf if minimum <= 0 else max(ka, kb) / minimum
    a_visual = a['KC_pattern_sha256'] != dark['KC_pattern_sha256']
    b_visual = b['KC_pattern_sha256'] != dark['KC_pattern_sha256']
    distinct = a['KC_pattern_sha256'] != b['KC_pattern_sha256']
    readout_available = sum(a['MBON']) > 0 and sum(b['MBON']) > 0
    balanced = ratio <= maximum_activity_ratio
    ready = a_visual and b_visual and distinct and readout_available and balanced
    return {
        'cue_a': a['label'],
        'cue_b': b['label'],
        'KC_a': ka,
        'KC_b': kb,
        'KC_activity_ratio': ratio,
        'cue_a_differs_from_dark': a_visual,
        'cue_b_differs_from_dark': b_visual,
        'KC_patterns_distinct': distinct,
        'MBON_readout_available': readout_available,
        'activity_balanced': balanced,
        'conditioning_ready': ready,
    }


def analyze_cues(rows, *, maximum_activity_ratio=2.0, legacy_pair=LEGACY_PAIR):
    if maximum_activity_ratio < 1 or not math.isfinite(maximum_activity_ratio):
        raise ValueError('maximum_activity_ratio must be finite and >= 1')
    by_label = {row['label']: row for row in rows}
    if 'black' not in by_label:
        raise ValueError('black control is required')
    dark = by_label['black']

    labels = sorted(label for label in by_label if label != 'black')
    pairs = []
    for i, left in enumerate(labels):
        for right in labels[i + 1:]:
            pairs.append(_pair_metrics(
                by_label[left], by_label[right], dark, maximum_activity_ratio
            ))
    pairs.sort(key=lambda item: (
        not item['conditioning_ready'],
        item['KC_activity_ratio'],
        item['cue_a'],
        item['cue_b'],
    ))

    legacy = None
    if all(label in by_label for label in legacy_pair):
        legacy = _pair_metrics(
            by_label[legacy_pair[0]], by_label[legacy_pair[1]],
            dark, maximum_activity_ratio,
        )

    ready_pairs = [item for item in pairs if item['conditioning_ready']]
    return {
        'maximum_activity_ratio': maximum_activity_ratio,
        'legacy_pair': legacy,
        'legacy_pair_ready': bool(legacy and legacy['conditioning_ready']),
        'ready_pair_count': len(ready_pairs),
        'recommended_pair': ready_pairs[0] if ready_pairs else None,
        'pairs': pairs,
    }
