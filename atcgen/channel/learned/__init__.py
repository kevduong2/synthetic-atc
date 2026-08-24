"""Mode 2: channel presets fitted to real calibration clips (04 §2).

`channel_fit` fits one preset per real clip offline (PyTorch, gradient
descent); `preset` holds the data format and the numpy evaluator that
generation uses; `backend` is the `ChannelBackend` that samples presets, real
noise and shared post-effects per utterance.

`residual_train` (M2.4) trains the optional CUT translator for the gap the fit
leaves, and `residual` applies it.  Neither is re-exported here: both import
torch, and the default generation path is numpy-only.
"""

from .backend import CalibratedChannel, StationNoise
from .preset import BAND_EDGES, Preset, apply_preset, load_presets, write_presets

__all__ = ["BAND_EDGES", "CalibratedChannel", "Preset", "StationNoise",
           "apply_preset", "load_presets", "write_presets"]
