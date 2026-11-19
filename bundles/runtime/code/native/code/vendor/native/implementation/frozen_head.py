"""Exact old head readout; H is a pure transform namespace, not model code."""
import numpy as np
from types import SimpleNamespace
from frozen_transform import transform
H=SimpleNamespace(transform=transform)

def predict_logits(cp, X):
    norm = {key: cp['scaler_' + key] for key in ('mean', 'scale', 'completely_absent_training_feature')}
    z = H.transform(X, norm)
    logits = np.tanh(z @ cp['hidden_weight'].T + cp['hidden_bias']) @ cp['output_weight'].T + cp['output_bias']
    if not np.isfinite(logits).all():
        raise FloatingPointError('Nonfinite fixed head output')
    return logits
