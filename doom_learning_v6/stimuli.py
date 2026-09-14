"""Deterministic visual stimuli for controlled v6 conditioning research.

New paired patterns use the same RGB value and the same number of illuminated
pixels; only spatial arrangement changes. This does not guarantee equal neural
activity, which is why every pair still has to pass ``cue_screen``.
"""
import numpy as np

from doom_learning_v2.vision import frame_for as legacy_frame_for

LEGACY_LABELS = {
    'black', 'blue', 'green', 'white', 'left_blue', 'right_blue',
    'vertical', 'horizontal',
}
MATCHED_SPATIAL_CUES = (
    'checker_a', 'checker_b', 'quadrants_a', 'quadrants_b',
)


def _blue(mask, width, height):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[mask, 2] = 255
    return frame


def frame_for(label, width=320, height=240):
    if label in LEGACY_LABELS:
        return legacy_frame_for(label, width=width, height=height)

    yy, xx = np.indices((height, width))
    if label in ('checker_a', 'checker_b'):
        # 40 px divides the canonical 320x240 frame exactly (8 x 6 cells), so
        # complementary phases have exactly equal illuminated area.
        phase = (xx // 40 + yy // 40) % 2
        mask = phase == (0 if label == 'checker_a' else 1)
        return _blue(mask, width, height)

    if label in ('quadrants_a', 'quadrants_b'):
        left = xx < width // 2
        top = yy < height // 2
        diagonal = left == top
        mask = diagonal if label == 'quadrants_a' else ~diagonal
        return _blue(mask, width, height)

    raise ValueError(f'Unknown research stimulus: {label}')


def pixel_energy(label, width=320, height=240):
    """Small provenance helper; not a neural-equivalence claim."""
    frame = frame_for(label, width, height)
    return {
        'illuminated_channel_sum': int(frame.astype(np.uint64).sum()),
        'nonzero_channel_values': int(np.count_nonzero(frame)),
        'shape': list(frame.shape),
    }
