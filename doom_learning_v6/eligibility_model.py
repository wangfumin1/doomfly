"""V6 neural/visual dynamics with the preserved kernel eligibility-LTD enabled.

This is a comparison model, not a claim of biological correctness. The native v6
kernel still contains the earlier KC eligibility -> PPL101-gated LTD mechanism,
but the normal v6 Python path deliberately disables that write and applies the
centered rate rule instead. Here we keep every other v6 component fixed and enable
that existing local kernel rule so the plasticity mechanism can be compared under
the same sensory and conditioning protocol.
"""
import math

import numpy as np

from .brain import MemoryBrain
from .visual import VisualMemoryBrain
from doom_learning.common import digest

MODEL = 'eligibility-ltd-v6'
PARAMETERS = {
    'eligibility_tau_ms': 1000.0,
    'eta_per_dan_event': 0.001,
    'minimum_efficacy_fraction': 0.1,
    'mechanism': (
        'KC spikes increment a lazily decayed presynaptic eligibility trace. '
        'PPL101 spikes depress eligible existing KC->MBON11 edges according to '
        'the reconstructed DAN-to-MBON anatomical gain. No new edges are added.'
    ),
    'limits': (
        'LTD-only comparison candidate. Eligibility constants and exponential '
        'update are model choices; this is not an exact reconstruction of fly '
        'molecular plasticity.'
    ),
}


class EligibilityStepMixin:
    """Replace only the v6 centered Python rule with the native eligibility rule."""

    plasticity_model = MODEL

    def step(self, luminance, duration_ms, *, learning=False, stimulation=None, lamina_bias=12.0):
        if not math.isfinite(duration_ms) or duration_ms <= 0:
            raise ValueError('Invalid duration')
        remaining = round(duration_ms / self.dt)
        if remaining < 1:
            raise ValueError('Duration too short')

        total = np.zeros(self.n, dtype=np.int32)
        wall = 0.0
        while remaining:
            ticks = min(100, remaining)
            interval = ticks * self.dt
            counts, elapsed = self._neural_step(
                luminance,
                interval,
                learning=bool(learning and not self.weights_frozen),
                stimulation=stimulation,
                lamina_bias=lamina_bias,
            )
            total += counts
            wall += elapsed
            remaining -= ticks
        self.counts[:] = total
        return total, wall

    def memory(self):
        weights = self.weight[self.circuit['edges']]
        fraction = weights / self.baseline_plastic
        return {
            'plastic_edges': len(weights),
            'changed_edges': int(np.count_nonzero(weights != self.baseline_plastic)),
            'mean_efficacy': float(fraction.mean()),
            'minimum_efficacy': float(fraction.min()),
            'maximum_efficacy': float(fraction.max()),
            'floor_fraction': PARAMETERS['minimum_efficacy_fraction'],
            'sha256': digest(weights),
            'model': MODEL,
        }

    def configuration_signature(self):
        return {
            **super().configuration_signature(),
            'plasticity_model': MODEL,
            'eligibility_model_parameters': PARAMETERS,
        }


class EligibilityMemoryBrain(EligibilityStepMixin, MemoryBrain):
    """Non-RGB form used by numerical/unit tests."""


class EligibilityVisualMemoryBrain(EligibilityStepMixin, VisualMemoryBrain):
    """Full v6 RGB model with native eligibility-LTD plasticity."""


def calibrated_brain(eta=0.001):
    """Use the exact v6 physiological background calibration for fair comparison."""
    brain = EligibilityVisualMemoryBrain(eta=eta)
    brain.tonic[brain.circuit['mb']] = 9.87
    brain.tonic[brain.circuit['dan']] = 11.3125
    brain.dan_baseline_hz[:] = 20.09  # retained in state/provenance; rule does not subtract it
    brain.calibration = {
        'MBON_current': 9.87,
        'DAN_current': 11.3125,
        'observed_MBON_hz': [37, 37],
        'observed_DAN_hz': [23, 19],
        'target_MBON_hz': 37.16625,
        'target_MBON_sd': 9.06014,
        'target_DAN_hz': 20.09,
        'target_DAN_sd': 4.30193,
        'evidence': 'https://doi.org/10.1038/s41586-024-07819-w',
        'source_data': 'Figure 1, panels d/e, n=20 flies per type',
        'plasticity_model': MODEL,
        'limits': (
            'Same v6 background-current calibration. The native eligibility rule '
            'responds to actual PPL101 spikes rather than baseline-subtracted rates.'
        ),
    }
    return brain
