"""Explicit factories for plasticity mechanisms sharing v6 neural/visual dynamics."""

MODELS = ('centered-v6', 'eligibility-ltd-v6')


def calibrated_brain(model='centered-v6', eta=0.001):
    if model == 'centered-v6':
        from .calibration import calibrated_brain as factory
        return factory(eta)
    if model == 'eligibility-ltd-v6':
        from .eligibility_model import calibrated_brain as factory
        return factory(eta)
    raise ValueError(f'Unknown model: {model}')


def model_metadata(model):
    if model == 'centered-v6':
        from .brain import MODEL, PARAMETERS
        return {'registry_name': model, 'implementation_model': MODEL, 'parameters': PARAMETERS}
    if model == 'eligibility-ltd-v6':
        from .eligibility_model import MODEL, PARAMETERS
        return {'registry_name': model, 'implementation_model': MODEL, 'parameters': PARAMETERS}
    raise ValueError(f'Unknown model: {model}')
