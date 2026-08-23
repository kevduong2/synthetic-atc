"""Mode 2: channel presets fitted to real calibration clips (04 §2).

`channel_fit` fits one preset per real clip offline (PyTorch, gradient
descent); `preset` holds the data format and the numpy evaluator that
generation uses; `backend` is the `ChannelBackend` that samples presets, real
noise and shared post-effects per utterance.
"""

from .backend import CalibratedChannel, StationNoise
from .preset import BAND_EDGES, Preset, apply_preset, load_presets, write_presets

__all__ = ["BAND_EDGES", "CalibratedChannel", "Preset", "StationNoise",
           "apply_preset", "load_presets", "write_presets"]
