"""Mock detectron2 for MaskFeat compatibility"""
__version__ = "0.6.mock"

class LazyConfig:
    """Minimal config stub"""
    pass

from . import config
from . import model_zoo
